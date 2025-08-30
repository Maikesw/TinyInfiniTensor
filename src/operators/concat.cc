#include "operators/concat.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业
    // ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业
    // ===================================

    auto n = inputs.size();
    for (size_t i = 1; i < n; i++) {
        auto input_dims = inputs[i]->getDims();
        if (input_dims.size() != rank) {
            return std::nullopt;
        }
        for (size_t j = 0; j < rank; j++) {
            if (j == size_t(dim)) {
                dims[j] += input_dims[j];
            } else if (dims[j] != input_dims[j]) {
                return std::nullopt;
            }
        }
    }
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
