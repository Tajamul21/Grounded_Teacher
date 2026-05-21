import csv
import sys

def filter_by_confidence(input_file, output_file, threshold):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                mean_conf = float(row['MeanConfidence'])
            except ValueError:
                continue  # skip rows with invalid or NA values

            if mean_conf >= threshold:
                writer.writerow(row)

    print(f"Done! Filtered rows written to '{output_file}' where MeanConfidence >= {threshold}")


if __name__ == "__main__":
    

    input_file = "/DATA/Tawheed/SFDA/Grounded_Teacher/Expert_Labels/brats_results.txt"
    output_file = "/DATA/Tawheed/SFDA/Grounded_Teacher/Expert_Labels/brats_results_filtered.txt"
    threshold = float(0.5)

    filter_by_confidence(input_file, output_file, threshold)
