# 2048-dqn

An attempt to slove 2048 by dqn

Implementation of 2048 is based on [ankitaggarwal011/2048-console](https://github.com/ankitaggarwal011/2048-console), MIT license

Modified for better performance. The biggest changes is to use int instead of str internally.

## Files

* 2048.py
    * Implementation of the game logic
    * Implementation of DQN using TensorFlow
    * Should work on both Python 2 and Python 3
* 2048-eval-http.py
    * A HTTP interface to utilize the trained model with other 2048 frontends
    * In Python 2
* 2048-eval-http-p3.py
    * 2048-eval-http, but in Python 3
* model.\*.ckpt.\*
    * Trained model

## Performance

_( As of last update to this doc )_

### model.1545678840.187.ckpt

Reaches about 20000 pts in average.

Biggest tile is 1024 most of the times, sometimes reaches 2048, small chance to reach 4096.

## Design

Train fixed times after each episode instead of each step to reduce training time and increase stability

### State

log2 of the table

### Reward

log2 of the addition to the score

## HTTP interface

_json pretty-printed for readability, but can be minified in practice_

Recieve POST with json representing the game table

```json
{
    "rows": [
        {
            "columns":[
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                }
            ]
        },
        {
            "columns":[
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                }
            ]
        },
        {
            "columns":[
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":2
                }
            ]
        },
        {
            "columns":[
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":0
                },
                {
                "val":2
                }
            ]
        }
    ]
}
```

Response with an action

```json
{
    "action":2
}
```

where 0, 1, 2, 3 are left, bottom, right, top respectively
