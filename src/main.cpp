#include "NeuralNetwork/matrix.hpp"
#include "NeuralNetwork/neuralnetwork.hpp"
#include "utils/utils.hpp"
#include "SFML/Graphics.hpp"

#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

vector<vector<int>> examples, test_examples;
vector<char> answer;
vector<vector<float>> inputs, outputs, targets;

void readData(const char filename[100]) {
    ifstream fin;
    fin.open(filename);

    if (!fin.is_open()) {
        cout << "Error open train file!\n";
        return;
    }

    string header;
    getline(fin, header);

    string line;
    int idx = 0;
    char c;
    while (!fin.eof()) {
        getline(fin, line);
        for (int i = 0; i < (int)line.size(); ++i) {
            if (line[i] == ',') line[i] = ' ';
        }
        stringstream buffer(line);
        
        answer.push_back('0');
        buffer >> answer[idx];

        examples.push_back(vector<int>(784, 0));
        for (int i = 0; i < 784; ++i) {
            buffer >> examples[idx][i];
        }

        ++idx;
    }

    fin.close();

    cout << "Done parsing train data!\n";
}

void parseTestData(const char filename[100]) {
    ifstream fin;
    fin.open(filename);

    if (!fin.is_open()) {
        cout << "Error open test file!\n";
        return;
    }

    string header;
    getline(fin, header);

    string line;
    int idx = 0;
    char c;
    while (!fin.eof()) {
        getline(fin, line);
        for (int i = 0; i < (int)line.size(); ++i) {
            if (line[i] == ',') line[i] = ' ';
        }
        stringstream buffer(line);

        test_examples.push_back(vector<int>(784, 0));
        for (int i = 0; i < 784; ++i) {
            buffer >> test_examples[idx][i];
        }

        ++idx;
    }

    fin.close();

    cout << "Done parsing test data!\n";
}

void shuffle(vector<int> &v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        swap(v[i], v[(int)floor(random(0, (float)v.size()))]);
    }
}

void draw(int testIdx) {
    // Convert pixels to image
    sf::Image image;
    image.create(28, 28);
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int index = y * 28 + x;
            sf::Uint8 value = static_cast<sf::Uint8>(test_examples[testIdx][index] * 255); // if normalized
            image.setPixel(x, y, sf::Color(value, value, value));
        }
    }

    // Display on windows
    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite(texture);
    sprite.setScale(10.0f, 10.0f); // scale 10x to make it 280×280

    sf::RenderWindow window(sf::VideoMode(280, 280), "Digit Recognizer");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
}

int main() {
    srand(time(0));
    
    readData("src/data/train.csv");

    NeuralNetwork nn(784, 128, 10);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        inputs.push_back(vector<float>(784, 0));
        outputs.push_back(vector<float>(10, 0));
        targets.push_back(vector<float>(10, 0));
    }

    cout << "Done initializing inputs, outputs, targets!\n";
    
    // Training
    vector<int> idx;
    for (int epoch = 1; epoch <= 10; ++epoch) {
        cout << "Training epoch " << epoch << '\n';
        // shuffle the data
        idx.clear();
        for (int i = 0; i < 42000; ++i) idx.push_back(i);
        shuffle(idx);

        // Traverse examples
        int cnt = 0;
        for (int i = 0; i < 42000; ++i) {
            // Prepare inputs, targets
            int index = idx[i];
            for (int j = 0; j < 10; ++j) {
                targets[cnt][j] = 0;
            }
            targets[cnt][answer[index] - '0'] = 1;

            for (int j = 0; j < 784; ++j) {
                inputs[cnt][j] = examples[index][j] / 255.0;
            }

            ++cnt;
            // Training a mini batch of size BATCH_SIZE
            if (cnt == BATCH_SIZE) {
                nn.MBGD_train(inputs, targets);
                cnt = 0;
            }
        }
        cout << "Finish epoch " << epoch << '\n';
    }
    cout << "Done training!\n";

    parseTestData("src/data/test.csv");

    // Testing
    int testIdx;
    for (testIdx = 0; testIdx < 18000; ++testIdx) {
        float test_inputs[784], test_outputs[10];
        for (int i = 0; i < 784; ++i) {
            test_inputs[i] = test_examples[testIdx][i];
        }
        nn.SGD_feedForward(test_inputs, test_outputs);

        for (int i = 0; i < 10; ++i) cout << fixed << setprecision(6) << test_outputs[i] << " ";
        cout << '\n';

        cout << "Predict: " << chosenDigit(test_outputs) << '\n';

        draw(testIdx);
    }

    return 0;
}