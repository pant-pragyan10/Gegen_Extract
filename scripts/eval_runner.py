from gegenextract.scoring.evaluator import Evaluator


def main():
    evalr = Evaluator()
    pred = {"name": "Alice", "items": [{"id": 1, "amt": 10}, {"id": 2, "amt": 20}]}
    gold = {"name": "Alice", "items": [{"id": 1, "amt": 10}, {"id": 2, "amt": 20}]}
    res = evalr.score(pred, gold)
    print(res)


if __name__ == "__main__":
    main()
