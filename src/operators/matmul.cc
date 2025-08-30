#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
       << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << ","
       << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    // =================================== 作业
    // ===================================
    // TODO：返回经过 matmul 操作后的 shape
    // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    // =================================== 作业
    // ===================================

    auto rankA = inputs[0]->getRank();
    auto rankB = inputs[1]->getRank();
    auto dimA = inputs[0]->getDims();
    auto dimB = inputs[1]->getDims();
    if (rankA < 2 || rankB < 2) {
        return {};
    }
    auto rankR = std::max(rankA, rankB);
    Shape res(rankR);
    int i = int(rankA) - 3, j = int(rankB) - 3;
    for (; i >= 0 && j >= 0; i--, j--) {
        if (dimA[i] == dimB[j] || dimB[j] == 1) {
            res[std::max(i, j)] = dimA[i];
        } else if (dimA[i] == 1) {
            res[std::max(i, j)] = dimB[j];
        } else {
            return {};
        }
    }
    for (; i >= 0; i--) {
        res[i] = dimA[i];
    }
    for (; j >= 0; j--) {
        res[j] = dimB[j];
    }
    auto m = dimA[rankA - 2], kA = dimA[rankA - 1], kB = dimB[rankB - 2],
         n = dimB[rankB - 1];
    if (this->transA) {
        std::swap(m, kA);
    }
    if (this->transB) {
        std::swap(kB, n);
    }
    if (kA != kB) {
        return {};
    }
    res[rankR - 2] = m;
    res[rankR - 1] = n;
    if (inputs.size() <= 2) {
        return {{res}};
    }
    auto rankC = inputs[2]->getRank();
    auto dimC = inputs[2]->getDims();
    if (rankC > rankR) {
        return {};
    }
    for (int i = rankC - 1, j = rankR - 1; i >= 0; i--, j--) {
        if (dimC[i] != res[j] && dimC[i] != 1) {
            return {};
        }
    }
    return {{res}};
}

} // namespace infini
