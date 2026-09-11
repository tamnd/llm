package prompt

import (
	"strings"
	"testing"
	"testing/fstest"
)

// The prompts here are short and made up. A prompt in a real set is prose a
// person maintains; what this package has to get right is the hashing, the
// naming and the refusal to send an unfilled placeholder.
func files() fstest.MapFS {
	return fstest.MapFS{
		"read.md":      &fstest.MapFile{Data: []byte("  Read the page.\n\n\n")},
		"translate.md": &fstest.MapFile{Data: []byte("Translate into {{LANGUAGE}}.\nKeep {{LANGUAGE}} terms as {{GLOSSARY}}.\n")},
	}
}

func TestGetTrimsAndHashes(t *testing.T) {
	set := New(files())
	got, err := set.Get("read")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Name != "read" {
		t.Errorf("name = %q, want the file name without its extension", got.Name)
	}
	// Two prompts differing only in how an editor left the last line must hash
	// the same, or a whitespace change restates a whole corpus.
	if got.Text != "Read the page.\n" {
		t.Errorf("text = %q", got.Text)
	}
	if got.SHA != SHA256("Read the page.\n") {
		t.Errorf("sha = %q", got.SHA)
	}
	if len(got.SHA) != 64 {
		t.Errorf("sha is %d characters, want the full hex", len(got.SHA))
	}
	// The extension is optional and the name is cached either way.
	same, err := set.Get("read.md")
	if err != nil || same.SHA != got.SHA {
		t.Errorf("Get with the extension = %q, %v", same.SHA, err)
	}
}

// The hash goes in the front matter of every file the prompt produced, so
// editing the prompt has to change it.
func TestEditingAPromptChangesItsHash(t *testing.T) {
	first := New(files()).MustGet("read")
	edited := files()
	edited["read.md"] = &fstest.MapFile{Data: []byte("Read the page carefully.\n")}
	if second := New(edited).MustGet("read"); second.SHA == first.SHA {
		t.Error("a rewritten prompt kept its hash, so its output would look current")
	}
}

func TestMissingPromptNamesWhatThereIs(t *testing.T) {
	set := New(files())
	_, err := set.Get("summarise")
	if err == nil {
		t.Fatal("an unknown prompt was returned")
	}
	if !strings.Contains(err.Error(), "translate") {
		t.Errorf("error = %q, want the available names in it", err)
	}
	if _, err := set.Get("  "); err == nil {
		t.Error("a prompt with no name was accepted")
	}
}

// Two files with one base name is a build mistake, and answering it with
// whichever the walk reached first would be a coin toss over what the binary
// sends.
func TestAmbiguousNameIsAnError(t *testing.T) {
	fsys := files()
	fsys["translate.txt"] = &fstest.MapFile{Data: []byte("other\n")}
	_, err := New(fsys).Get("translate")
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("err = %v, want it to refuse to choose", err)
	}
}

func TestMustGetPanicsOnAMissingPrompt(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a missing embedded prompt was survivable")
		}
	}()
	New(files()).MustGet("nothing")
}

func TestNamesAndAll(t *testing.T) {
	set := New(files())
	names := set.Names()
	if len(names) != 2 || names[0] != "read" || names[1] != "translate" {
		t.Errorf("names = %v, want both in order", names)
	}
	all, err := set.All()
	if err != nil || len(all) != 2 {
		t.Fatalf("All = %d prompts, %v", len(all), err)
	}
	for _, p := range all {
		if p.SHA == "" || p.Text == "" {
			t.Errorf("prompt %s came back unfilled", p.Name)
		}
	}
}

func TestRenderFillsEveryOccurrence(t *testing.T) {
	p := New(files()).MustGet("translate")
	out, err := p.Render(map[string]string{"LANGUAGE": "Vietnamese", "GLOSSARY": "given"})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if strings.Contains(out, "{{") {
		t.Errorf("rendered text still has a placeholder in it: %q", out)
	}
	if strings.Count(out, "Vietnamese") != 2 {
		t.Errorf("rendered = %q, want both occurrences filled", out)
	}
	// A caller rendering three prompts from one map is doing something
	// reasonable, so a spare variable is not an error.
	if _, err := p.Render(map[string]string{"LANGUAGE": "Chinese", "GLOSSARY": "g", "SPARE": "x"}); err != nil {
		t.Errorf("a spare variable was refused: %v", err)
	}
}

// A prompt sent with a literal {{GLOSSARY}} in it is an instruction the model
// cannot follow, and the answers come back subtly worse with nothing saying
// why.
func TestRenderRefusesAnUnfilledPlaceholder(t *testing.T) {
	p := New(files()).MustGet("translate")
	_, err := p.Render(map[string]string{"LANGUAGE": "Japanese"})
	if err == nil {
		t.Fatal("an unfilled prompt was rendered")
	}
	if !strings.Contains(err.Error(), "GLOSSARY") {
		t.Errorf("error = %q, want the missing name", err)
	}
	// Named once, even though the template could use it twice.
	if strings.Count(err.Error(), "GLOSSARY") != 1 {
		t.Errorf("error = %q, want each missing name once", err)
	}
}

func TestVarsAreInFirstAppearanceOrder(t *testing.T) {
	p := New(files()).MustGet("translate")
	vars := p.Vars()
	if len(vars) != 2 || vars[0] != "LANGUAGE" || vars[1] != "GLOSSARY" {
		t.Errorf("vars = %v", vars)
	}
	// Lower case braces are not placeholders: prose and code samples in a
	// prompt are full of them.
	plain := Prompt{Text: "Leave {this} and {{lower}} alone.\n"}
	if got := plain.Vars(); len(got) != 0 {
		t.Errorf("vars = %v, want none", got)
	}
	if out, err := plain.Render(nil); err != nil || out != plain.Text {
		t.Errorf("Render = %q, %v, want the text untouched", out, err)
	}
}
