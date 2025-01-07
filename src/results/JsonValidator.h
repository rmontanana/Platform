#ifndef JSONVALIDATOR_H
#define JSONVALIDATOR_H
#include <fstream>
#include <vector>
#include <regex>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    class JsonValidator {
    public:
        JsonValidator(const json& schema) : schema(schema) {}

        std::vector<std::string> validate(const std::string& fileName)
        {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                return { "Error: Unable to open file." };
            }

            json data;
            try {
                file >> data;
            }
            catch (const json::parse_error& e) {
                return { "Error: JSON parsing failed: " + std::string(e.what()) };
            }

            std::vector<std::string> errors;

            // Validate all fields defined in the schema
            for (const auto& [key, value] : schema["properties"].items()) {
                validateField(key, value, data, errors);
            }

            return errors;
        }

        void fix_it(const std::string& fileName)
        {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                std::cerr << "Error: Unable to open file for fixing." << std::endl;
                return;
            }

            json data;
            try {
                file >> data;
            }
            catch (const json::parse_error& e) {
                std::cerr << "Error: JSON parsing failed: " << e.what() << std::endl;
                return;
            }
            file.close();

            // Fix fields
            for (const auto& [key, value] : schema["properties"].items()) {
                if (!data.contains(key)) {
                    // Set default value if specified in the schema
                    if (value.contains("default")) {
                        data[key] = value["default"];
                    } else if (value["type"] == "array") {
                        data[key] = json::array();
                    } else if (value["type"] == "object") {
                        data[key] = json::object();
                    } else {
                        data[key] = nullptr;
                    }
                }

                // Fix const fields to match the schema value
                if (value.contains("const")) {
                    data[key] = value["const"];
                }
            }

            // Save fixed JSON
            std::ofstream outFile(fileName);
            if (!outFile.is_open()) {
                std::cerr << "Error: Unable to open file for writing." << std::endl;
                return;
            }

            outFile << data.dump(4);
            outFile.close();
        }
    private:
        json schema;

        void validateField(const std::string& field, const json& value, const json& data, std::vector<std::string>& errors)
        {
            // Check if the field is present
            if (!data.contains(field)) {
                errors.push_back("Missing required field: " + field);
                return;
            }

            // Check for type constraints
            if (value.contains("type")) {
                const std::string type = value["type"];
                if (type == "string" && !data[field].is_string()) {
                    errors.push_back("Field '" + field + "' should be a string.");
                } else if (type == "number" && !data[field].is_number()) {
                    errors.push_back("Field '" + field + "' should be a number.");
                } else if (type == "integer" && !data[field].is_number_integer()) {
                    errors.push_back("Field '" + field + "' should be an integer.");
                } else if (type == "boolean" && !data[field].is_boolean()) {
                    errors.push_back("Field '" + field + "' should be a boolean.");
                } else if (type == "array" && !data[field].is_array()) {
                    errors.push_back("Field '" + field + "' should be an array.");
                } else if (type == "object" && !data[field].is_object()) {
                    errors.push_back("Field '" + field + "' should be an object.");
                }
            }

            // Check for const constraints
            if (value.contains("const")) {
                const auto& expectedValue = value["const"];
                if (data[field] != expectedValue) {
                    errors.push_back("Field '" + field + "' has an invalid value. Expected: " +
                        expectedValue.dump() + ", Found: " + data[field].dump());
                }
            }
        }

    };
}
#endif