converter = {
            "pink": 0,
            "white": 1,
            "plastic": 0,
            "glass": 1,
            "0": 0,
            "10": 1,
            "15": 2,
            "20": 3,
            "min": 0
            }

decode = {
    0: {
        0: "pink",
        1: "white",
    },
    1: {
        0: "plastic",
        1: "glass",
    },
    2: {
        0: "0",
        1: "10",
        2: "15",
        3: "20",
    }
}

task_types = {
    "champagne_type": 1,
    "container_type": 2,
    "time": 3
}