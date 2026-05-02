import csv
from collections import defaultdict
import statistics

input_file = 'batch_eval_all12.csv'
output_file = 'batch_eval_summary.csv'

# Columns to calculate mean and std for
numeric_cols = [
    'max_capacity_bits', 'actual_bpp', 'bits_embedded', 
    'chars_embedded', 'mse', 'psnr_db', 'ssim', 'RS analysis'
]

# group data by target_bpp
# data[target_bpp][col_name] = [value1, value2, ...]
data = defaultdict(lambda: defaultdict(list))

with open(input_file, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            target_bpp = float(row['target_bpp'])
        except ValueError:
            continue
        
        for col in numeric_cols:
            if col in row:
                try:
                    val = float(row[col])
                    data[target_bpp][col].append(val)
                except ValueError:
                    pass

# Calculate mean and sd and write to new csv
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    # Prepare header
    header = ['target_bpp']
    for col in numeric_cols:
        header.append(f'{col}_mean')
        header.append(f'{col}_sd')
    
    writer = csv.writer(f)
    writer.writerow(header)
    
    # Write rows sorted by target_bpp
    for target_bpp in sorted(data.keys()):
        row = [target_bpp]
        for col in numeric_cols:
            values = data[target_bpp].get(col, [])
            if not values:
                row.extend(['', ''])
            else:
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                row.extend([mean, sd])
        writer.writerow(row)

print(f"Summary saved to {output_file}")
