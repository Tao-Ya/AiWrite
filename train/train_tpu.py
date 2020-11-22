# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Training script! """

import tensorflow as tf

import sys
sys.path.append("..")

from train.dataloader import input_fn_builder
from train.modeling import model_fn_builder, GroverConfig

flags = tf.flags

FLAGS = flags.FLAGS

import os.path as op
path = op.dirname(op.dirname(op.abspath(__file__)))

## Required parameters
# 配置json文件对应于预先训练好的新闻模型
flags.DEFINE_string(
    "config_file", path + '/configs/base.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

# 输入TF示例文件(可以是通配符或逗号分隔)
flags.DEFINE_string(
    "input_file", path + "/dataset/wiki_train_wiki19_0000.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

# 将要写入模型检查点的输出目录
flags.DEFINE_string(
    "output_dir", path + "/models/wiki",
    "The output directory where the model checkpoints will be written.")

## Other parameters
# 初始检查点(通常来自预先训练好的模型)
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained model).")

# BPE标记化后输入序列的最大总长度
# 超过这个长度的序列会被截断，而长度会更短
# 这将被填充。必须匹配数据生成。
flags.DEFINE_integer(
    "max_seq_length", 1024,
    "The maximum total input sequence length after BPE tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

# 培训的bs值
# flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

# 学习率
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for adafactor.")

# 训练步数
# flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")
flags.DEFINE_integer("num_train_steps", 100, "Number of training steps.")

# 热训练步数
# flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")
flags.DEFINE_integer("num_warmup_steps", 10, "Number of warmup steps.")

# 多久保存一次模型的检查点
# flags.DEFINE_integer("save_checkpoints_steps", 1000,
#                      "How often to save the model checkpoint.")
flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

# 在每个估计器调用中要做多少步
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

# eval步骤的最大数目
flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

# 使用TPU/GPU/CPU
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# 配置云TPU信息
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

# 云TPU所在的[可选]GCE区域，如果没有指定，我们将尝试从“metadata”自动检测GCE项目
flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# 支持云tpu的项目的[可选]项目名称。如果没有指定，我们将尝试从元数据中自动检测GCE项目
flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# 可选TensorFlow的地址
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# 仅在use_tpu为真时使用，要使用的TPU核总数。
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    news_config = GroverConfig.from_json_file(FLAGS.config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(news_config, init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=FLAGS.num_train_steps,
                                num_warmup_steps=FLAGS.num_warmup_steps,
                                use_tpu=FLAGS.use_tpu,
                                )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.train_batch_size,
        params={'model_dir': FLAGS.output_dir}
    )

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        seq_length=FLAGS.max_seq_length,
        is_training=True)

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
