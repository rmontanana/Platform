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

        std::vector<std::string> validate_file(const std::string& fileName)
        {
            auto data = load_json_file(fileName);
            return validate(data);
        }
        std::vector<std::string> validate(const json& data)
        {
            std::vector<std::string> errors;
            // Validate the top-level object
            validateObject("", schema, data, errors);
            return errors;
        }
        json load_json_file(const std::string& fileName)
        {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                throw std::runtime_error("Error: Unable to open file " + fileName);
            }
            json data;
            file >> data;
            file.close();
            return data;
        }
        void fix_it(const std::string& fileName)
        {
            // Load JSON file
            auto data = load_json_file(fileName);
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

        void validateObject(const std::string& path, const json& schema, const json& data, std::vector<std::string>& errors)
        {
            if (schema.contains("required")) {
                for (const auto& requiredField : schema["required"]) {
                    if (!data.contains(requiredField)) {
                        std::string fullPath = path.empty() ? requiredField.get<std::string>() : path + "." + requiredField.get<std::string>();
                        errors.push_back("Missing required field: " + fullPath);
                    }
                }
            }
            if (schema.contains("properties")) {
                for (const auto& [key, value] : schema["properties"].items()) {
                    if (data.contains(key)) {
                        std::string fullPath = path.empty() ? key : path + "." + key;
                        validateField(fullPath, value, data[key], errors);  // Pass data[key] for nested validation
                    } else if (value.contains("required")) {
                        errors.push_back("Missing required field: " + (path.empty() ? key : path + "." + key));
                    }
                }
            }
        }

        void validateField(const std::string& field, const json& value, const json& data, std::vector<std::string>& errors)
        {
            if (value.contains("type")) {
                const std::string& type = value["type"];
                if (type == "array") {
                    if (!data.is_array()) {
                        errors.push_back("Field '" + field + "' should be an array.");
                        return;
                    }

                    if (value.contains("items")) {
                        for (size_t i = 0; i < data.size(); ++i) {
                            validateObject(field + "[" + std::to_string(i) + "]", value["items"], data[i], errors);
                        }
                    }
                } else if (type == "object") {
                    if (!data.is_object()) {
                        errors.push_back("Field '" + field + "' should be an object.");
                        return;
                    }

                    validateObject(field, value, data, errors);
                } else if (type == "string" && !data.is_string()) {
                    errors.push_back("Field '" + field + "' should be a string.");
                } else if (type == "number" && !data.is_number()) {
                    errors.push_back("Field '" + field + "' should be a number.");
                } else if (type == "integer" && !data.is_number_integer()) {
                    errors.push_back("Field '" + field + "' should be an integer.");
                } else if (type == "boolean" && !data.is_boolean()) {
                    errors.push_back("Field '" + field + "' should be a boolean.");
                }
            }
            if (value.contains("const")) {
                const auto& expectedValue = value["const"];
                if (data != expectedValue) {
                    errors.push_back("Field '" + field + "' has an invalid value. Expected: " +
                        expectedValue.dump() + ", Found: " + data.dump());
                }
            }
        }
    };
}
#endif