#pragma once
#include "converters.hpp"
#include <fstream>
#include <random>
#include <stdexcept>
#include <unordered_map>

pair<vector<vector<any>>, vector<any>> Xy_from_csv(const string &file_path, const int y_idx = -1, const bool header = false) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Could not open file: " << file_path << endl;
        throw runtime_error("Could not open file: " + file_path);
    }
    string line;
    if (!header) getline(file, line); // Skip header

    vector<vector<any>> X_raw;
    vector<any> y_raw;
    unordered_map<string, int> class_to_index = {};
    int class_index = 0;

    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<any> row;
        size_t start = 0, end = 0;
        while ((end = line.find(',', start)) != string::npos) {
            string cell_value = line.substr(start, end - start);
            row.push_back(str_to_any(cell_value));
            start = end + 1;
        }
        row.push_back(str_to_any(line.substr(start))); // Add the last cell

        int target_col = y_idx < 0 ? row.size() + y_idx : y_idx;                          // Negative index means counting from the end
        X_raw.emplace_back(row.begin(), row.begin() + target_col);                        // Initialize with all columns except the target
        X_raw.back().insert(X_raw.back().end(), row.begin() + target_col + 1, row.end()); // Append the rest of the columns

        if (row[target_col].type() == typeid(string)) {
            const string &class_label = any_cast<string>(row[target_col]); // & is necessary to avoid copying
            if (class_to_index.find(class_label) == class_to_index.end())  // Not found
                class_to_index[class_label] = class_index++;
            y_raw.push_back(class_to_index[class_label]);
        } else y_raw.push_back(row[target_col]);
    }
    return {X_raw, y_raw};
}

pair<vector<vector<any>>, vector<any>> shuffle_data(const vector<vector<any>> &X, const vector<any> &y) {
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., X.size() - 1
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

    vector<vector<any>> X_shuffled;
    vector<any> y_shuffled;
    for (int idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
    }
    return {X_shuffled, y_shuffled};
}

tuple<vector<vector<any>>, vector<vector<any>>, vector<any>, vector<any>>
train_test_split(const vector<vector<any>> &X, const vector<any> &y, float test_size = 0.2) {
    pair<vector<vector<any>>, vector<any>> shuffled_data = shuffle_data(X, y);
    vector<vector<any>> X_shuffled = shuffled_data.first;
    vector<any> y_shuffled = shuffled_data.second;

    size_t train_size = X.size() * (1 - test_size);
    vector<vector<any>> X_train(X_shuffled.begin(), X_shuffled.begin() + train_size);
    vector<vector<any>> X_test(X_shuffled.begin() + train_size, X_shuffled.end());

    vector<any> y_train(y_shuffled.begin(), y_shuffled.begin() + train_size);
    vector<any> y_test(y_shuffled.begin() + train_size, y_shuffled.end());
    return {X_train, X_test, y_train, y_test};
}

class StandardScaler {
private:
    vector<double> means, stds;

public:
    vector<vector<double>> fit_transform(const vector<vector<any>> &X) {
        int n_samples = X.size(), n_features = X[0].size();
        means.resize(n_features, 0.0);
        stds.resize(n_features, 0.0);

        // Calculate mean and std for each feature
        for (int f = 0; f < n_features; ++f) {
            for (const vector<any> &sample : X)
                means[f] += any_to_double(sample[f]);
            means[f] /= n_samples;

            for (const vector<any> &sample : X)
                stds[f] += pow(any_to_double(sample[f]) - means[f], 2);
            stds[f] = sqrt(stds[f] / n_samples);
            if (stds[f] == 0) stds[f] = 1.0; // Prevent division by 0
        }
        return transform(X);
    }

    vector<vector<double>> transform(const vector<vector<any>> &X) {
        int n_samples = X.size(), n_features = X[0].size();
        vector<vector<double>> X_scaled(n_samples, vector<double>(n_features));

        for (int i = 0; i < n_samples; ++i)
            for (int f = 0; f < n_features; ++f)
                X_scaled[i][f] = (any_to_double(X[i][f]) - means[f]) / stds[f];
        return X_scaled;
    }
};