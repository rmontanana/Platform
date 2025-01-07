#ifndef SCHEMAV1_0_H
#define SCHEMAV1_0_H
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    class SchemaV1_0 {
    public:
        // Define JSON schema
        const static json schema;

    };
    const json SchemaV1_0::schema = {
        {"$schema", "http://json-schema.org/draft-07/schema#"},
        {"type", "object"},
        {"properties", {
            {"schema_version", {
                {"type", "string"},
                {"pattern", "^\\d+\\.\\d+$"},
                {"default", "1.0"},
                {"const", "1.0"}  // Fixed schema version for this schema
            }},
            {"date", {{"type", "string"}, {"format", "date"}}},
            {"time", {{"type", "string"}, {"pattern", "^\\d{2}:\\d{2}:\\d{2}$"}}},
            {"title", {{"type", "string"}}},
            {"language", {{"type", "string"}}},
            {"language_version", {{"type", "string"}}},
            {"discretized", {{"type", "boolean"}, {"default", false}}},
            {"model", {{"type", "string"}}},
            {"platform", {{"type", "string"}}},
            {"stratified", {{"type", "boolean"}, {"default", false}}},
            {"folds", {{"type", "integer"}, {"default", 0}}},
            {"score_name", {{"type", "string"}}},
            {"version", {{"type", "string"}}},
            {"duration", {{"type", "number"}, {"default", 0}}},
            {"results", {
                {"type", "array"},
                {"items", {
                    {"type", "object"},
                    {"properties", {
                        {"scores_train", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                        {"scores_test", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                        {"times_train", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                        {"times_test", {{"type", "array"}, {"items", {{"type", "number"}}}}},
                        {"notes", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                        {"train_time", {{"type", "number"}, {"default", 0}}},
                        {"train_time_std", {{"type", "number"}, {"default", 0}}},
                        {"test_time", {{"type", "number"}, {"default", 0}}},
                        {"test_time_std", {{"type", "number"}, {"default", 0}}},
                        {"samples", {{"type", "integer"}, {"default", 0}}},
                        {"features", {{"type", "integer"}, {"default", 0}}},
                        {"classes", {{"type", "integer"}, {"default", 0}}},
                        {"hyperparameters", {
                            {"type", "object"},
                            {"additionalProperties", {
                                {"oneOf", {
                                    {{"type", "number"}},  // Field can be a number
                                    {{"type", "string"}}   // Field can also be a string
                                }}
                            }}
                        }},
                        {"score", {{"type", "number"}, {"default", 0}}},
                        {"score_train", {{"type", "number"}, {"default", 0}}},
                        {"score_std", {{"type", "number"}, {"default", 0}}},
                        {"score_train_std", {{"type", "number"}, {"default", 0}}},
                        {"time", {{"type", "number"}, {"default", 0}}},
                        {"time_std", {{"type", "number"}, {"default", 0}}},
                        {"nodes", {{"type", "number"}, {"default", 0}}},
                        {"leaves", {{"type", "number"}, {"default", 0}}},
                        {"depth", {{"type", "number"}, {"default", 0}}},
                        {"dataset", {{"type", "string"}}},
                        {"confusion_matrices", {
                            {"type", "array"},
                            {"items", {
                                {"type", "object"},
                                {"patternProperties", {
                                    {".*", {
                                        {"type", "array"},
                                        {"items", {{"type", "integer"}}}
                                    }}
                                }},
                                {"additionalProperties", false}
                            }}
                        }}
                    }},
                    {"required", {
                        "scores_train", "scores_test", "times_train", "times_test",
                        "train_time", "train_time_std", "test_time", "test_time_std",
                        "samples", "features", "classes", "hyperparameters", "score", "score_train",
                        "score_std", "score_train_std", "time", "time_std", "nodes", "leaves",
                        "depth", "dataset"
                    }}
                }}
            }}
        }},
        {"required", {
            "schema_version", "date", "time", "title", "language", "language_version",
            "discretized", "model", "platform", "stratified", "folds", "score_name",
            "version", "duration", "results"
        }}
    };
}
#endif