# llm

Go library for running long jobs across a mixed fleet of language models: a subscription CLI on this machine, browser session proxies on rented boxes, free OpenAI compatible gateways, and a GPU box serving its own weights.

It is the transport, routing, queueing and record keeping that [tamnd/bourbaki-solver](https://github.com/tamnd/bourbaki-solver) grew over a 1699 page corpus, pulled out so that the next project does not grow it again.

Standard library only, no dependencies, and CI fails if that ever stops being true.

## Install

```sh
go get github.com/tamnd/llm
```

## The shape of the problem

A round trip to a browser session proxy is around 150 seconds.
A free gateway answers a 429 with `retry-after: 53931`, which is fifteen hours.
A subscription runs out of turns until Tuesday.
A box's signed in sessions get banned in batches.
Of 17,123 asks across one week, between a third and a half of each day was lost before the question was read at all.

Nothing here assumes API latency.
Work is a durable on disk queue with leases, routes fail over by cost and cool down by cause, and every ask is recorded so that "this route is configured" and "this route is producing answers" stay different claims.

## Packages

| package | what it is |
| --- | --- |
| `llm` | the client: one OpenAI compatible `Complete`, streaming, images, and the classifier that turns a status and a body into a state and a time |
| `route` | the route file, the five kinds of route, and the pool that picks one, fails over, and cools down by cause |
| `queue` | a directory of jobs with leases, content addressed ids, bounded attempts and crash recovery |
| `prompt` | prompts as embedded files, hashed, with `{{NAME}}` substitution that refuses to send an unfilled placeholder |
| `exec` | a route that is a program on this machine rather than an endpoint somewhere else |
| `fleet` | ssh to the boxes, tunnels to their loopback listeners, and one readiness question per host |
| `ledger` | one JSONL line per ask, appended and never rewritten |

## Configure it

The library carries no application name, so two tools on one machine do not share a config directory or an environment variable by accident.

```go
llm.Configure(llm.Config{App: "papers"})
```

That decides `~/.config/papers/`, the `PAPERS_*` environment prefix falling back to `LLM_*`, and the user agent.

## Routes

Five kinds, because they fail differently, and a caller that cannot tell them apart cannot fail over sensibly.

| kind | what it is | how it fails |
| --- | --- | --- |
| `pool` | a browser session proxy over an ssh tunnel | sessions banned, held, or signed out |
| `gateway` | a free OpenAI compatible endpoint | quota with hours on it, models withdrawn |
| `exec` | a subscription reached through its own CLI | out of turns |
| `reader` | a GPU box serving its own weights, which answers images rather than questions | the model server is down |
| `direct` | a local vLLM or ollama | it is not running |

The route file lives at `~/.config/<app>/routes.json` and is the caller's own.
Nothing in this repository names a host, a port, a key or a model.
`route.Default()` is empty, `route.Suggest()` is unfilled templates, keys are read from environment variables named by the file, and a literal key never round trips back into it.

```go
registry, err := route.Load(route.DefaultPath())
pool := route.NewPool(registry)
pool.Build = exec.Build // so an exec route can be built too, not only an HTTP one

picked, client, release, err := pool.Pick(ctx)
if err != nil {
    return err // every route is cold, and the error says when the first comes back
}
defer release()

answer, err := client.Complete(ctx, llm.Request{Instructions: instructions, Input: text})
if err != nil {
    pool.Fail(picked.Name, err) // classified, and cooled down by cause
    return err
}
pool.Succeed(picked.Name)
```

`Pick` takes the cheapest route that is live and holds a lane on it.
On failure it classifies what came back: a spent quota cools the route down until it resets, a rejected key for six hours, a withdrawn model retires it, and a transport error backs off from a minute.
A 429 carrying fifteen hours means go to the next route, not sleep.

## Queue

```go
q, err := queue.Open(root, "read", "translate")
q.Add(queue.New("translate", "secA/001", inputSHA, prompts.MustGet("translate").SHA))

job, err := q.Lease("translate", host, "secA", 5*time.Minute)
state, err := q.Finish(job, ok, reason)
```

The id is the content address of the work, meaning the stage, the target, the input hash and the prompt hash together.
A rerun of the same pipeline therefore adds nothing, and an edited prompt is new work rather than a repeat of old work.
The claim is a rename, which settles a race between two processes or two machines sharing a directory.
A lease is not a lock: a worker that dies leaves a job whose deadline passes, and any worker starting up puts it back.

## Testing

`go test ./...` needs no network, no subscription and no box.
Every transport is an interface or a function field, every clock is injected, and the recorded fixtures are invented text in the shape the real thing prints.

## Licence

MIT.
