import cotk.dataloader
import json

def run():
    dataloader = cotk.dataloader.MSCOCO("resources://MSCOCO_small")
    metric = dataloader.get_inference_metric()
    metric.forward({
        "gen":
            [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
            [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    })
    json.dump(metric.close(), open("result.json", 'w'))
