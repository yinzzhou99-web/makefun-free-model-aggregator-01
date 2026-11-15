# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/yinzzhou99-web/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/yinzzhou99-web/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator server for make.fun: it serves OpenAI- and Anthropic-compatible APIs over ten simulated free-tier providers, ranking every free model that meets a request's requirements by the quality table's rating, failing over on refusals, speaking all three provider wires (OpenAI, Anthropic, Gemini), and returning each answer in the client's wire, streamed or not. Local practice on the kit's three trials: 20 of 20 valid, score 99 on each.

## Run it

```sh
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=fp85vftq&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
