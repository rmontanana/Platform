#ifndef DELONG_H
#define DELONG_H
/* ********************************************************************************************************************
/* Integración del test de DeLong con la clase RocAuc y Statistics
/* Basado en: X. Sun and W. Xu, "Fast Implementation of DeLong’s Algorithm for Comparing the Areas Under Correlated
/* Receiver Operating Characteristic Curves," (2014), y algoritmos inspirados en sklearn/pROC
/* ********************************************************************************************************************/
#include <vector>

namespace platform {
    class DeLong {
    public:
        struct DeLongResult {
            double auc_diff;
            double z_stat;
            double p_value;
        };
        // Compara dos vectores de AUCs por dataset y devuelve diferencia media,
        // estadístico z y p-valor usando un test de rangos (DeLong simplificado)
        static DeLongResult compare(const std::vector<double>& aucs_model1,
            const std::vector<double>& aucs_model2);
    };
}
#endif // DELONG_H