from dspy.dsp.utils.settings import settings


def get_traces(named_predicts):
    predict_name_to_traces = {}
    predict_id_to_name = {id(predict): name for name, predict in named_predicts.items()}

    traces = settings.trace
    for i in range(len(traces)):
        trace = traces[-i - 1]
        trace_predict_id = id(trace[0])
        if trace_predict_id in predict_id_to_name:
            predict_name = predict_id_to_name[trace_predict_id]
            if predict_name not in predict_name_to_traces:
                predict_name_to_traces[predict_name] = {
                    "inputs": trace[1],
                    "outputs": trace[2].toDict(),
                }
        if len(predict_name_to_traces) == len(named_predicts):
            # Stop searching when all predicts' traces are found.
            break
    return predict_name_to_traces
