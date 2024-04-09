#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
#include </home/czhang/OpenSfM/bsiCPP/bsi/BsiAttribute.hpp> // TODO: change to relative path
#include </home/czhang/OpenSfM/bsiCPP/bsi/BsiSigned.hpp>
#include </home/czhang/OpenSfM/bsiCPP/bsi/BsiUnsigned.hpp>
namespace py = pybind11;
/*template <typename T>
class Bsi {
public:
    std::vector<int> shape;
    Bsi() {}
    Bsi(py::list<int> array) {
        shape = {py::len(array)};
        BsiSigned<uint64_t> bsi;
        BsiAttribute<uint64_t>* bsiattribute = bsi.buildBsiAttributeFromVector(py::cast<std::vector<int>>array);
    }
    T arr;
}*/
/*std::vector<long> listToVec(py::list a) {
    std::vector<long> res;
    for (py::handle item: a) {
        try {
            res.push_back(item.cast<long>());
        } catch (std::exception e) {throw std::runtime_error("could not cast to type long");}
    }
    return res;
}
long dot(py::list a, py::list b) {
    BsiSigned<uint64_t> bsi;
    BsiAttribute<uint64_t> *bsi_a = bsi.buildBsiAttributeFromVectorSigned(listToVec(a),0.5);
    BsiAttribute<uint64_t> *bsi_b = bsi.buildBsiAttributeFromVectorSigned(listToVec(a),0.5);
    //BsiAttribute<uint64_t> *bsi_a = bsi.buildBsiAttributeFromVectorSigned(py::cast<std::vector<long>> (a),0.5);
    //BsiAttribute<uint64_t> *bsi_b = bsi.buildBsiAttributeFromVectorSigned(py::cast<std::vector<long>> (a),0.5);
    return bsi_a->dot(bsi_b);
}
PYBIND11_MODULE(pybsi,m) {
    //py::class_<Bsi<BsiAttribute<u_int64_t>>>(m,"bsi")
    //    .def(py::init<>());
    m.def("dot", &dot);
}*/
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(pybsi, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}