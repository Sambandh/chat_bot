import os
from gpt_2_simple.src import model, sample, encoder
# import fire
from transformer import  *
import json
import numpy as np
import tensorflow as tf
import gpt_2_simple as gpt2

print('start#############')
model_name = "124M"

gpt2.download_gpt2(model_name=model_name)
tf.compat.v1.reset_default_graph()
sess = gpt2.start_tf_sess()
gpt2.finetune(sess, file_name, model_name=model_name, steps=20)

def interact_model(model_name, seed, nsmaples, batch_size, length, temperature, top_k, models_dir):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsmaples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Cant fetch samples longer thab window size"% hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(hparams=hparams, length=length, context=context, batch_size=batch_size, temperature=temperature, top_k=top_k)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>>>>")
            while not raw_text:
                print ('Prompt should not be empty!')
                raw_text = input("Model prompt >>>")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsmaples // batch_size):
                out = sess.run(output, feed_dict={context: [context_tokens for _ in range(batch_size)]})[:, len(context_tokens):]
                for i in range(batch_size):
                    generated=generated+1
                    text = enc.encode(out[i])
                    print("=" * 40 + "SAMPLE" + str(generated)+ " " + "=" * 40)
                    print (text)
                print ("=" * 80)

interact_model('run1',None,1,1,2,1,0,'./checkpoint')

