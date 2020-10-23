import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.

from absl import app  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module

def find_op(einsum_str,inputs,max_solutions=1,time_limit=300,n_samples=1):
  #print("Warm up...")
  colab_interface.warm_up()
  #print("Warm up done")
  #if n_samples == 1 :
  outputs = tf.einsum(einsum_str,*inputs)
  #else :
  #  g_inputs = [inputs]
  #  for i in range(n_samples):
  #   g_inputs.append( #Tensor With Same Shape )
  # outputs = [ tf.einsum(einsum_str,*inp)  for inp in g_inputs ]

  settings = settings_module.from_dict({
      'timeout': time_limit,
      'only_minimal_solutions': False,
      'max_solutions': max_solutions,
      'require_all_inputs_used': True,
      'require_one_input_used': False,
  })

  colab_interface.run_value_search_from_colab(inputs,output,[],"",settings)


if __name__ == '__main__':
  find_op('i,i->',[
        tf.constant([1,2,3]) , tf.constant([4,5,6])
   ])
  #app.run(main)
