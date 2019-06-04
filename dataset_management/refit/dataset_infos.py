

aggregate_mean = 522
aggregate_std = 814

params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        'test_house': 2,
        'validation_house': 5,
        'test_on_train_house': 5,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [4, 10, 12, 17, 19],
        'channels': [8, 8, 3, 7, 4],
        'test_house': 4,
        'validation_house': 17,
        'test_on_train_house': 10,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [2, 5, 9, 12, 15],
        'channels': [1, 1, 1,  1, 1],
        'test_house': 15,
        'validation_house': 12,
        'test_on_train_house': 5,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [5, 7, 9, 13, 16, 18, 20],
        'channels': [4, 6, 4, 4, 6, 6, 5],
        'test_house': 9,
        'validation_house': 18,
        'test_on_train_house': 13,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
        'test_house': 8,
        'validation_house': 18,
        'test_on_train_house': 5,
    }
}


columns_names = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
houses = {
    '1': [10, 10, 1, 10, 5],
    '2': [8, 5, 1, 10, 2],
    '3': [9, 8, 2, 10, 6],
    '4': [9, 8, 1, 4, 6],  # 4 and 6 washingmachines
    '5': [8, 7, 1, 4, 3],
    '6': [7, 6, 1, 3, 2],
    '7': [9, 10, 1, 6, 5],
    '8': [9, 8, 1, 10, 4],
    '9': [7, 6, 1, 4, 3],
    '10': [10, 8, 4, 6, 5],
    '12': [6, 5, 1, 10, 10],
    #'13': [9, 8, 4, 6, 5],
    '15': [10, 7, 1, 4, 3],
    '16': [10, 10, 1, 6, 5],
    '17': [8, 7, 2, 10, 4],
    '18': [10, 9, 1, 6, 5],
    '19': [5, 4, 1, 10, 2],
    '20': [9, 8, 1, 5, 4],
}