// Package prompt holds prompts as files rather than as string literals.
//
// Files, so that a prompt can be read and edited as prose by somebody who is
// not reading Go. Embedded, so that a binary carries the exact text it was
// built with and not whatever is on the disk beside it. Hashed, so that the
// hash can go into the front matter of every file the prompt produced: when
// the prompt changes, the output it produced is detectably stale rather than
// silently mixed with output produced by a different one.
//
// A caller embeds its own directory of prompts and hands the filesystem over:
//
//	//go:embed *.md
//	var files embed.FS
//	var Prompts = prompt.New(files)
//
// The library it was extracted from had a function per prompt — OCR(),
// Translate(lang) — which is a corpus's API dressed as a library's. Prompts
// are the caller's; the mechanism is not.
package prompt

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
)

// Prompt is one prompt, with the hash of its template.
type Prompt struct {
	// Name is the file name without its extension: "translate", not
	// "translate.md".
	Name string
	Text string
	// SHA is the sha256 of Text in full hex, which is what goes in an output
	// file's prompt_sha256.
	//
	// It is over the template and not over the rendered text. The rendered
	// text varies per chunk and hashing that would make every chunk its own
	// prompt; the template is the thing whose change should invalidate a
	// corpus.
	SHA string
}

// Set is a directory of prompts.
type Set struct {
	fsys fs.FS
	mu   sync.Mutex
	seen map[string]Prompt
}

// New wraps a filesystem, usually an embed.FS.
func New(fsys fs.FS) *Set { return &Set{fsys: fsys, seen: map[string]Prompt{}} }

// Get reads a prompt by name, with or without its extension. The text is
// trimmed and given a single trailing newline, so that two prompts differing
// only in how an editor left the last line hash the same.
func (s *Set) Get(name string) (Prompt, error) {
	key := strings.TrimSpace(name)
	if key == "" {
		return Prompt{}, fmt.Errorf("prompt has no name")
	}
	s.mu.Lock()
	if cached, ok := s.seen[key]; ok {
		s.mu.Unlock()
		return cached, nil
	}
	s.mu.Unlock()

	file, err := s.resolve(key)
	if err != nil {
		return Prompt{}, err
	}
	raw, err := fs.ReadFile(s.fsys, file)
	if err != nil {
		return Prompt{}, fmt.Errorf("read prompt %s: %w", file, err)
	}
	text := strings.TrimSpace(string(raw)) + "\n"
	value := Prompt{
		Name: strings.TrimSuffix(path.Base(file), path.Ext(file)),
		Text: text,
		SHA:  SHA256(text),
	}
	s.mu.Lock()
	s.seen[key] = value
	s.mu.Unlock()
	return value, nil
}

// MustGet is Get for a prompt the program cannot run without. It panics,
// because a missing embedded prompt is a build mistake and not a runtime
// condition: the file is either in the binary or it never was.
func (s *Set) MustGet(name string) Prompt {
	value, err := s.Get(name)
	if err != nil {
		panic(err)
	}
	return value
}

// resolve finds the file for a name, preferring an exact match and otherwise
// taking the one file whose base name matches. Two files with the same base
// and different extensions are an error rather than a coin toss.
func (s *Set) resolve(name string) (string, error) {
	if _, err := fs.Stat(s.fsys, name); err == nil {
		return name, nil
	}
	all, err := s.files()
	if err != nil {
		return "", err
	}
	var matches []string
	for _, file := range all {
		if strings.TrimSuffix(path.Base(file), path.Ext(file)) == name {
			matches = append(matches, file)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return "", fmt.Errorf("no prompt named %q, have %s", name, strings.Join(s.Names(), ", "))
	default:
		return "", fmt.Errorf("prompt %q is ambiguous: %s", name, strings.Join(matches, ", "))
	}
}

func (s *Set) files() ([]string, error) {
	var out []string
	err := fs.WalkDir(s.fsys, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			out = append(out, p)
		}
		return nil
	})
	sort.Strings(out)
	return out, err
}

// Names lists every prompt in the set, for an error message and for a command
// that prints what is available.
func (s *Set) Names() []string {
	files, err := s.files()
	if err != nil {
		return nil
	}
	out := make([]string, 0, len(files))
	for _, file := range files {
		out = append(out, strings.TrimSuffix(path.Base(file), path.Ext(file)))
	}
	slices.Sort(out)
	return slices.Compact(out)
}

// All reads every prompt in the set, for a command that prints the hashes.
func (s *Set) All() ([]Prompt, error) {
	files, err := s.files()
	if err != nil {
		return nil, err
	}
	out := make([]Prompt, 0, len(files))
	for _, file := range files {
		value, err := s.Get(file)
		if err != nil {
			return out, err
		}
		out = append(out, value)
	}
	return out, nil
}

// placeholder matches {{NAME}}: upper case, digits and underscores.
var placeholder = regexp.MustCompile(`\{\{([A-Z][A-Z0-9_]*)\}\}`)

// Render substitutes {{NAME}} from vars and fails on a placeholder nothing
// filled.
//
// The failing is the point and it is not what the original did. A prompt sent
// with a literal {{GLOSSARY}} still in it is a prompt the model reads as an
// instruction it cannot follow, and the answers come back subtly worse with
// nothing in any log saying why. A missing variable is a programming mistake
// and should read like one.
//
// A variable in vars that the template does not use is allowed: a caller that
// renders three prompts from one map is doing something reasonable.
func (p Prompt) Render(vars map[string]string) (string, error) {
	var missing []string
	out := placeholder.ReplaceAllStringFunc(p.Text, func(match string) string {
		name := match[2 : len(match)-2]
		value, ok := vars[name]
		if !ok {
			missing = append(missing, name)
			return match
		}
		return value
	})
	if len(missing) > 0 {
		slices.Sort(missing)
		missing = slices.Compact(missing)
		return "", fmt.Errorf("prompt %s has no value for %s", p.Name, strings.Join(missing, ", "))
	}
	return out, nil
}

// Vars lists the placeholders a template uses, in first-appearance order, so
// a caller can check a map before a run rather than at the first chunk.
func (p Prompt) Vars() []string {
	var out []string
	seen := map[string]bool{}
	for _, match := range placeholder.FindAllStringSubmatch(p.Text, -1) {
		if !seen[match[1]] {
			seen[match[1]] = true
			out = append(out, match[1])
		}
	}
	return out
}

// SHA256 hashes a prompt, in full hex.
func SHA256(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}
