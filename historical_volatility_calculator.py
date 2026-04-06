import argparse
import csv
import statistics

# Expects the csv to be in the form: stock price, date of price
def main():
    # Manage cli arguments and flags
    parser = argparse.ArgumentParser(description="A simple CLI app to calculate the historical volatility of a given stock.")

    parser.add_argument("stocks_filepath", help="The csv file for the stock to analyze.")
    parser.add_argument("-a", "--analyze", type=float, default=0.5, help="Specify the percentage of stock data in the file to be analyzed for variance.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold to determine high volatility.")

    args = parser.parse_args()

    # Extract and transform data
    data = extract_data(args.stocks_filepath)

    analysis_split = int(len(data) * args.analyze)

    analysis_data = data[:analysis_split]
    inference_data = data[analysis_split:]

    threshold = calculate_volatility(analysis_data, args.threshold)

    print("High-Volatility Threshold: (>95%)")
    print(f"Upper bound: {threshold['upper_bound']:.2f}")
    print(f"Lower bound: {threshold['lower_bound']:.2f}\n\n")

    # Determine volatility
    print("Measuring Volatility:")
    for sample in inference_data:
        if sample["price"] <= threshold["lower_bound"] or sample["price"] >= threshold["upper_bound"]:
            print(sample["date"], "High")
        else:
            print(sample["date"], "Low")


def calculate_volatility(data, threshold):
    data = [x["price"] for x in data]
    avg = statistics.mean(data)
    std_dev = statistics.stdev(data)

    return {"lower_bound": avg - (2 * std_dev), "upper_bound": avg + (2 * std_dev)}


def extract_data(stocks_filepath):
    data = []
    with open(stocks_filepath, mode="r", newline='') as file:
        reader = csv.reader(file)

        header = next(reader)
        # print(f"Headers: {header}")

        for row in reader:
            data.append({"price": float(row[0]), "date": row[1]})
            
    return data

if __name__ == "__main__":
    main()