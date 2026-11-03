# TODOS

Design notes raised in review on 2026-10-31 and deliberately not fixed yet.

- **Repair metric can be gamed.** `repair_candidate` is "a rejected request on a doable task". A policy that fires a deliberate throwaway bad request and then succeeds counts as a repair. Before training, tighten the definition to "a rejected request to endpoint X followed by an accepted request to X", or report both. The summary now splits success by drifted versus clean episodes, which is the honest number until then.
- **Sequential ids let a batch guess instead of read.** The mock server allocates ids as max+1, so a model can batch `POST /books` and `POST /books/4/checkout` in one turn and be rewarded without reading the response. Decide whether ids should be non-sequential in generated worlds.
- **The task instruction is inserted into the prompt raw.** Facts are JSON-encoded because they come from an orchestrator; the instruction will too, and it is not escaped. Decide, when the orchestrator is wired, whether the instruction is trusted or gets the same treatment.
- **Server responses are echoed into `<response>` blocks unescaped.** A model-chosen string such as a title containing `</response>` cannot fool the parser (it is JSON-side) but can confuse the model's own context. Escape or note before training.
- **No fail-fast on model/config errors in the eval loop.** An authentication failure produces one error row per task. Abort after the first error of that class.
- **Frontier models fabricate `<response>` blocks.** In the 2026-10-31 live run Haiku wrote its own server response inside a request turn on 3 of 9 tasks; the parser now rejects that with a specific message, which means the earlier 9/9 baseline partly rewarded hallucinated responses. Re-run the baseline and watch the parse-failure rate on small models.
- **Parser feedback edge cases.** Prose after the last block that mentions `<request>` is reported as truncation; a literal `<request>` inside a request body string is reported as invalid JSON. Both cost a parse failure. Decide on a neutral message or an outermost-opener regex.
- **Integer coercion is Python's `int()`.** `/books/+1`, `/books/ 1` and full-width digits are accepted. Use a `^-?\d+$` check if "strict validation" is to mean strict.
