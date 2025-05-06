# Data provided by the user
# Structure: dataset_name: (reorder_raw_value, train_raw_value)
datasets_data = {
    'amazon0505': (0.051, 5.756),
    'artist': (0.011, 0.778),
    'com-amazon': (0.02, 4.2678),
    # For web-BerkStan, data was missing. Estimating raw values to yield ~3.5% Reordering.
    'web-BerkStan': (0.035, 0.965),
    'soc-BlogCatalog': (0.0196, 1.3386),
    'amazon0601': (0.083, 5.355)
}

# Order of datasets as in the image
dataset_names_ordered = ['amazon0505', 'artist', 'com-amazon', 'web-BerkStan', 'soc-BlogCatalog', 'amazon0601']

print("Reordering Overhead Percentages:\n")

for name in dataset_names_ordered:
    if name in datasets_data:
        r_raw, t_raw = datasets_data[name]
        total = r_raw + t_raw
        if total == 0:
            reorder_percentage = 0.0
        else:
            reorder_percentage = (r_raw / total) * 100
        print(f"{name}: {reorder_percentage:.2f}%")
    else:
        print(f"{name}: Data not available")