import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle
from time import time

DEBUG = False


class MonitorOperator(mx.operator.CustomOp):
    def __init__(self, nickname):
        super(MonitorOperator, self).__init__()
        self.nickname= nickname

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        if 'start' not in globals().keys():
            globals()['start'] = []
            globals()['end'] = []
        if self.nickname == 'resnet_start':
            globals()['start'].append(time())
        else:
            globals()['end'].append(time())
            #assert len(globals()['end']) == len(globals()['start']), 'end length {} doesnt equal to start length {}'.format(len(globals()['end']), len(globals()['start']))
            if len(globals()['end']) > 0 and len(globals()['end']) % 20 == 0:
                offset = len(globals()['start']) - len(globals()['end'])
                if offset == 0:
                    end_time = (np.array(globals()['end'])[-20:] - np.array(globals()['start'])[-20:]).mean()
                else:
                    end_time = (np.array(globals()['end'])[-20:] - np.array(globals()['start'])[-20 - offset:-offset]).mean()
                print("average network forward time: {}".format(end_time))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register('monitor')
class MonitorProp(mx.operator.CustomOpProp):
    def __init__(self, nickname):
        super(MonitorProp, self).__init__(need_top_grad=False)
        self.nickname = nickname

    def list_arguments(self):
        return ['input']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return [output_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MonitorOperator(self.nickname)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]


def monitor_wrapper(sym_instance, name):
    return mx.sym.Custom(input=sym_instance,
                         op_type='monitor',
                         nickname=name)

