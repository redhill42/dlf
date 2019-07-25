#include "model.h"
#include "onnx.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace dlf { namespace model {
namespace {

using namespace onnx;

// Part 1: convert ONNX Protobuf to IR

std::unique_ptr<Graph> decodeGraph(const GraphProto& gp, bool nested);

TensorData decodeTensor(const TensorProto& tp) {
    TensorData ret;

    ret.set_name(tp.name());
    ret.set_type(static_cast<DataType>(tp.data_type()));
    ret.set_dims({tp.dims().begin(), tp.dims().end()});

    switch (tp.data_type()) {
    case TensorProto::FLOAT:
    case TensorProto::COMPLEX64:
        ret.float_data().assign(tp.float_data().begin(), tp.float_data().end());
        break;

    case TensorProto::FLOAT16:
    case TensorProto::BFLOAT16:
    case TensorProto::BOOL:
    case TensorProto::INT8:
    case TensorProto::INT16:
    case TensorProto::INT32:
    case TensorProto::UINT8:
    case TensorProto::UINT16:
        ret.int32_data().assign(tp.int32_data().begin(), tp.int32_data().end());
        break;

    case TensorProto::INT64:
        ret.int64_data().assign(tp.int64_data().begin(), tp.int64_data().end());
        break;

    case TensorProto::UINT32:
    case TensorProto::UINT64:
        ret.uint64_data().assign(tp.uint64_data().begin(), tp.uint64_data().end());
        break;

    case TensorProto::DOUBLE:
    case TensorProto::COMPLEX128:
        ret.double_data().assign(tp.double_data().begin(), tp.double_data().end());
        break;

    case TensorProto::STRING:
        ret.string_data().assign(tp.string_data().begin(), tp.string_data().end());
        break;

    default:
        fail_convert("Unknown tensor data type ", tp.data_type());
    }

    if (tp.has_raw_data())
        ret.set_raw_data(tp.raw_data());

    return ret;
}

void decodeAttribute(const AttributeProto& ap, Node* n) {
    auto name = Symbol(ap.name());

    switch (ap.type()) {
    case AttributeProto::FLOAT:
        n->set_f(name, ap.f());
        break;

    case AttributeProto::INT:
        n->set_i(name, ap.i());
        break;

    case AttributeProto::STRING:
        n->set_s(name, ap.s());
        break;

    case AttributeProto::TENSOR:
        n->set_t(name, decodeTensor(ap.t()));
        break;

    case AttributeProto::GRAPH:
        n->set_g(name, decodeGraph(ap.g(), true));
        break;

    case AttributeProto::FLOATS:
        n->set_fs(name, {ap.floats().begin(), ap.floats().end()});
        break;

    case AttributeProto::INTS:
        n->set_is(name, {ap.ints().begin(), ap.ints().end()});
        break;

    case AttributeProto::STRINGS:
        n->set_ss(name, {ap.strings().begin(), ap.strings().end()});
        break;

    case AttributeProto::TENSORS: {
        std::vector<TensorData> ts;
        ts.reserve(ap.tensors_size());
        for (auto& tp : ap.tensors())
            ts.push_back(decodeTensor(tp));
        n->set_ts(name, std::move(ts));
        break;
    }

    case AttributeProto::GRAPHS: {
        std::vector<std::shared_ptr<Graph>> gs;
        gs.reserve(ap.graphs_size());
        for (auto& gp : ap.graphs())
            gs.push_back(decodeGraph(gp, true));
        n->set_gs(name, std::move(gs));
        break;
    }

    default:
        fail_convert("Unknown attribute data type ", ap.type());
        break;
    }
}

Dims decodeTensorShape(const TensorShapeProto& tsp) {
    Dims dims;
    for (auto& d : tsp.dim()) {
        if (d.has_dim_value()) {
            dims.append(static_cast<size_t>(d.dim_value()));
        } else {
            dims.append(d.dim_param());
        }
    }
    return dims;
}

void setValueType(Value* v, const TypeProto_Tensor& tp) {
    v->set_type(static_cast<DataType>(tp.elem_type()));
    if (tp.has_shape()) {
        v->set_dims(decodeTensorShape(tp.shape()));
    }
}

std::unique_ptr<Graph> decodeGraph(const GraphProto& gp, bool nested) {
    auto g = std::make_unique<Graph>();

    if (gp.has_name())
        g->set_name(gp.name());
    if (gp.has_doc_string())
        g->set_doc_string(gp.doc_string());

    // Values are created (as in `new Value(..)`) by the Node that
    // outputs them. Therefore we initialize the Nodes and Values in
    // several stages.
    //
    // 1) add all input (to the graph) Values, owned by the sentinel Param node
    // 2) add all Nodes and their output Values, but don't initialize inputs
    // 3) initialize inputs of all Nodes
    // 4) initialize inputs of the Return sentinel node
    // 5) fill in type info for graph outputs, and register them as outputs
    // 6) fill in type info for Values from the value_info list in the graph

    // In ONNX proto land, Values are just strings. We are going to make
    // objects out of them, and equal strings must be mapped to the same
    // Value object.
    std::unordered_map<std::string, Value*> value_by_name;

    // We initialize Node inputs in a separate pass from the Nodes
    // themselves. To do so, we need to have access to the names of the
    // inputs.
    std::unordered_map<Node*, std::vector<std::string>> inputs_by_node;

    // ONNX represents optional arguments in two ways
    //  - they are simply not provided
    //  - OR the empty string is passed as the input name
    // This is to handle that second case, which need a dummy node to
    // be representable in the graph IR.
    value_by_name[""] = g->undefinedValue();

    // Adding all inputs with type definition.
    for (auto& vp : gp.input()) {
        auto v = g->addInput(vp.name());
        setValueType(v, vp.type().tensor_type());
        value_by_name[vp.name()] = v;
    }

    // Adding all initializers with type and data.
    for (auto& init : gp.initializer()) {
        if (!init.has_name()) {
            std::cerr << "The initializer must have a name\n";
            continue;
        }
        auto t = decodeTensor(init);
        auto it = value_by_name.find(t.name());
        if (it == value_by_name.end()) {
            auto v = g->addInput(t.name(), t.type(), t.dims());
            it = value_by_name.emplace(t.name(), v).first;
        }
        it->second->set_initializer(std::move(t));
    }

    // Adding all nodes, defer determine value types of inputs and outputs.
    for (auto& np : gp.node()) {
        auto* n = g->appendNode(g->createNode(np.op_type()));

        for (auto& output : np.output()) {
            // we don't know the real type here, so that's done in a later pass
            value_by_name[output] = n->addOutput(output);
        }

        for (auto& ap : np.attribute()) {
            decodeAttribute(ap, n);
        }

        // we will connect inputs to other nodes' output later, we just
        // record input names now.
        inputs_by_node[n] = {np.input().begin(), np.input().end()};

        if (np.has_name())
            n->set_name(np.name());
        if (np.has_domain())
            n->set_domain(np.domain());
        if (np.has_doc_string())
            n->set_doc_string(np.doc_string());
    }

    // Connect node's inputs to other nodes' output.
    for (auto n : g->nodes()) {
        auto search = inputs_by_node.find(n);
        if (search != inputs_by_node.end()) {
            for (auto& input : search->second) {
                if (!value_by_name.count(input) && nested) {
                    // Undefined reference to an input in a nested block. This may be
                    // a captured value. Create a dummy node that we ignore later.
                    auto* undef = g->appendNode(g->createNode(kCaptured));
                    value_by_name[input] = undef->addOutput(input);
                }
                n->addInput(value_by_name[input]);
            }
        }
    }

    // Fill in value type from output definition
    for (auto& vp : gp.output()) {
        if (!value_by_name.count(vp.name()) && nested) {
            // Same captured value logic as above. We can consider outputs of
            // a graph to be "inputs" of a dummy "output" node. The same lexical
            // scoping rule are valid here, thus we need to add a dummy node
            // in the case of the undefined reference
            auto* undef = g->appendNode(g->createNode(kCaptured));
            value_by_name[vp.name()] = undef->addOutput(vp.name());
        }

        setValueType(value_by_name[vp.name()], vp.type().tensor_type());
        g->addOutput(value_by_name[vp.name()]);
    }

    // Fill in value type from value_info definition
    for (auto& vp : gp.value_info()) {
        setValueType(value_by_name[vp.name()], vp.type().tensor_type());
    }

    return g;
}

} // end of anonymous namespace

template <>
std::unique_ptr<Graph> import_model<ONNX>(std::istream& input) {
    ::google::protobuf::io::IstreamInputStream stream(&input);
    ::google::protobuf::io::CodedInputStream decoder(&stream);
    decoder.SetTotalBytesLimit(500*1024*1024, 64*1024*1024);

    ModelProto mp;
    if (!mp.ParseFromCodedStream(&decoder)) {
        fail_convert("Failed to parse model protocol");
    }

    return decodeGraph(mp.graph(), false);
}

// Part 2: convert IR to ONNX Protobuf

namespace {

void encodeGraph(GraphProto* gp, const Graph* g);

void encodeTensor(TensorProto* tp, const TensorData& t) {
    if (t.has_name())
        tp->set_name(t.name());
    for (auto d : t.dims())
        tp->add_dims(d);
    tp->set_data_type(static_cast<int32_t>(t.type()));

    switch(t.type()) {
    case DataType::FLOAT:
    case DataType::COMPLEX64:
        for (float x : t.float_data())
            tp->add_float_data(x);
        break;

    case DataType::FLOAT16:
    case DataType::BOOL:
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::UINT8:
    case DataType::UINT16:
        for (int32_t x : t.int32_data())
            tp->add_int32_data(x);
        break;

    case DataType::INT64:
        for (int64_t x : t.int64_data())
            tp->add_int64_data(x);
        break;

    case DataType::UINT32:
    case DataType::UINT64:
        for (uint64_t x : t.uint64_data())
            tp->add_uint64_data(x);
        break;

    case DataType::DOUBLE:
        for (double x : t.double_data())
            tp->add_double_data(x);
        break;

    case DataType::STRING:
        for (auto& x : t.string_data())
            tp->add_string_data(x);
        break;

    default:
        fail_convert("Unknown tensor data type");
    }

    if (t.has_raw_data()) {
        tp->set_raw_data(t.raw_data());
    }
}

void encodeAttribute(NodeProto* np, const Node* n, Symbol name) {
    auto ap = np->add_attribute();
    ap->set_name(name.str());

    switch (n->attributeKind(name)) {
    case AttributeKind::FLOAT:
        ap->set_type(AttributeProto::FLOAT);
        ap->set_f(n->get_f(name));
        break;

    case AttributeKind::FLOATS:
        ap->set_type(AttributeProto::FLOATS);
        for (auto v : n->get_fs(name))
            ap->add_floats(v);
        break;

    case AttributeKind::INT:
        ap->set_type(AttributeProto::INT);
        ap->set_i(n->get_i(name));
        break;

    case AttributeKind::INTS:
        ap->set_type(AttributeProto::INTS);
        for (auto v : n->get_is(name))
            ap->add_ints(v);
        break;

    case AttributeKind::STRING:
        ap->set_type(AttributeProto::STRING);
        ap->set_s(n->get_s(name));
        break;

    case AttributeKind::STRINGS:
        ap->set_type(AttributeProto::STRINGS);
        for (auto& v : n->get_ss(name))
            ap->add_strings(v);
        break;

    case AttributeKind::TENSOR:
        ap->set_type(AttributeProto::TENSOR);
        encodeTensor(ap->mutable_t(), n->get_t(name));
        break;

    case AttributeKind::TENSORS:
        ap->set_type(AttributeProto::TENSORS);
        for (auto& v : n->get_ts(name))
            encodeTensor(ap->add_tensors(), v);
        break;

    case AttributeKind::GRAPH:
        ap->set_type(AttributeProto::GRAPH);
        encodeGraph(ap->mutable_g(), n->get_g(name).get());
        break;

    case AttributeKind::GRAPHS:
        ap->set_type(AttributeProto::GRAPHS);
        for (auto& v : n->get_gs(name))
            encodeGraph(ap->add_graphs(), v.get());
        break;

    default:
        fail_convert("Unknown attribute type");
    }
}

void encodeValueInfo(ValueInfoProto* vp, const Value* v) {
    auto tp = vp->mutable_type()->mutable_tensor_type();
    vp->set_name(v->name());
    tp->set_elem_type(static_cast<int32_t>(v->type()));
    TensorShapeProto* shape = tp->mutable_shape();
    for (auto& d : v->dims()) {
        if (d.has_value())
            shape->add_dim()->set_dim_value(d.value());
        else
            shape->add_dim()->set_dim_param(d.symbol());
    }
}

void encodeGraph(GraphProto* gp, const Graph* g) {
    if (g->has_name())
        gp->set_name(g->name());
    if (g->has_doc_string())
        gp->set_doc_string(g->doc_string());

    for (auto input : g->inputs()) {
        encodeValueInfo(gp->add_input(), input);
        if (input->has_initializer()) {
            encodeTensor(gp->add_initializer(), input->initializer());
        }
    }

    for (auto output : g->outputs()) {
        encodeValueInfo(gp->add_output(), output);
    }

    std::unordered_set<const Value*> graph_outputs(g->outputs().begin(), g->outputs().end());

    for (auto node : g->nodes()) {
        if (node->kind() == kUndefined || node->kind() == kCaptured) {
            // Undefined nodes are used to represent optional inputs that are not provided.
            continue;
        }

        auto np = gp->add_node();
        np->set_op_type(node->kind().str());
        if (node->has_name())
            np->set_name(node->name());
        if (node->has_domain())
            np->set_domain(node->domain());
        if (node->has_doc_string())
            np->set_doc_string(node->doc_string());
        for (auto attr_name : node->attributeNames())
            encodeAttribute(np, node, attr_name);

        for (auto input : node->inputs()) {
            if (input->node()->kind() == kUndefined) {
                np->add_input("");
            } else {
                np->add_input(input->name());
            }
        }

        for (auto output : node->outputs()) {
            np->add_output(output->name());
            // only save it if
            //  - it has actual information worth saving
            //  - it's not already saved in the graph outputs value info
            if (graph_outputs.find(output) != graph_outputs.end())
                continue;
            if (output->type() == DataType::UNDEFINED)
                continue;
            encodeValueInfo(gp->add_value_info(), output);
        }
    }
}

} // end of anonymous namespace

template <>
void export_model<ONNX>(std::ostream& os, const Graph& g) {
    ModelProto mp;
    encodeGraph(mp.mutable_graph(), &g);
    if (!mp.SerializeToOstream(&os)) {
        fail_convert("Failed to serialize model");
    }
}

}} // namespace dlf::model
