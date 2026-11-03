# Cross-model comparison

### exp1 — which game was this network trained on?

| model | result |
| --- | --- |
| Astra (post) | 4/4 |
| deepseek-v4.1-flash | 3/4 (miss: pacman->asterix) |
| glm-5.3-flash | 1/4 (miss: breakout->None, pong->freeway, pacman->pong) |
| muse-spark-1.3-contributor | 3/4 (miss: freeway->None) |

### exp2 — what does this diffusion model generate?

| model | result |
| --- | --- |
| Astra (post) | 3/3 |
| deepseek-v4.1-flash | 1/3 (miss: square->circle, triangle->None) |
| glm-5.3-flash | 1/3 (miss: square->None, triangle->circle) |
| muse-spark-1.3-contributor | 0/3 (miss: circle->None, square->None, triangle->None) |

### exp3 — which language, and which heads copy?

| model | result |
| --- | --- |
| Astra (post) | 6/6 |
| deepseek-v4.1-flash | corpus 0/6, copy-set 0/6 |
| deepseek-v4.1-flash-xhigh | corpus 0/6, copy-set 0/6 |
| muse-spark-1.3-contributor | corpus 0/6, copy-set 0/6 |

### exp4 — fix a failing policy by editing its weights

| model | result |
| --- | --- |
| Astra (post) | 1024/1024 |
| deepseek-v4.1-flash | 0.0% success (1024 eps) |
| glm-5.3-flash | not scored |
| muse-spark-1.3-contributor | 0.0% success (256 eps) |

### exp5 — write game-playing networks from scratch

| model | result |
| --- | --- |
| Astra (post) | 517.4 / 53.1 / 53.2 |
| deepseek-v4.1-flash | breakout=8.3 / seaquest=0.0 / asterix=3.8 |
| deepseek-v4.1-flash-xhigh | breakout=45.9 / seaquest=0.2 / asterix=1.6 |
| glm-5.3-flash | breakout=0.0 / seaquest=0.5 / asterix=0.5 |
| muse-spark-1.3-contributor | breakout=9.6 / seaquest=0.5 / asterix=2.8 |

### exp6 — write a Go network from scratch

| model | result |
| --- | --- |
| Astra (post) | 0.426 / 0.263 / 0.363 |
| deepseek-v4.1-flash | 0.083 / 0.053 / 0.095 |
| glm-5.3-flash | 0.021 / 0.018 / 0.020 |
| muse-spark-1.3-contributor | 0.021 / 0.018 / 0.020 |

### exp7 — which language model is sycophantic?

| model | result |
| --- | --- |
| Astra (post) | attention 3/3 |
| attention:deepseek-v4.1-flash | 0/1 |
| attention:glm-5.3-flash | 0/1 |
| attention:muse-spark-1.3-contributor | 0/1 |
| weights_only:deepseek-v4.1-flash | 0/4 |
| weights_only:muse-spark-1.3-contributor | 0/4 |
