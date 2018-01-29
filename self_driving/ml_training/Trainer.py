def trainer(graph, config, model):
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    iterations = config["iterations"]
    learning_rate = config["learning_rate"]
    path_data = config["path_data"]
    path_labels = config["path_labels"]
    height = config["height"]
    width = config["width"]
    channels = config["channels"]
    tfrecords_name = config["tfrecords_name"]
    show_step = 10
    
    best_valid_loss = float("inf")
    
    records = data2record(path_data,
                          path_labels,
                          tfrecords_name,
                          height,
                          width,
                          channels)

    
    tfrecords_train = records[0]
    tfrecords_valid = records[1]
    tf_optimizer = config["optimizer"]
 

    with graph.as_default():
        
        input_image = tf.placeholder(tf.float32,
                                     shape=(None,
                                            height * width * channels),
                                    name="input_image")
        
        iterator_train = get_iterator(tfrecords_train,
                                batch_size,
                                parser_with_normalization)
        iterator_valid = get_iterator(tfrecords_valid,
                                     batch_size,
                                     parser_with_normalization)
        train_images, train_labels = iterator_train.get_next()
        train_images = tf.reshape(train_images, (batch_size, height * width * channels))
        train_labels = tf.reshape(train_labels, (batch_size,))
        train_logits = model.get_logits(train_images)
        tf_train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels,
                                                                   logits=train_logits)
        tf_train_loss = tf.reduce_mean(tf_train_loss)
        optimizer = tf_optimizer(learning_rate)
        update_weights = optimizer.minimize(tf_train_loss)

        valid_images, valid_labels = iterator_valid.get_next()
        valid_images = tf.reshape(valid_images, (batch_size, height * width * channels))
        valid_labels = tf.reshape(valid_labels, (batch_size,))
        valid_logits = model.get_logits(valid_images)
        valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels,
                                                                   logits=valid_logits)
        tf_valid_loss = tf.reduce_mean(valid_loss)
        
        tf_prediction = network.get_prediction(input_image)
        
        saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'best_validation')
        
        
        

    with tf.Session(graph=graph) as sess:
        sess.run(iterator_train.initializer)
        sess.run(iterator_valid.initializer)
        sess.run(tf.global_variables_initializer())
        show_loss = sess.run(tf_train_loss)
        print(sess.run(valid_loss))
        for i in range(epochs):
            print("loss =", show_loss)
            for step in range(iterations):
                _, my_loss = sess.run([update_weights, tf_train_loss]) 
                show_loss = my_loss
                if step % show_step == 0:
                    valid_loss = sess.run(tf_valid_loss)
                    if valid_loss < best_valid_loss:
                        saver.save(sess=sess,
                                   save_path=save_path)
                        best_valid_loss = valid_loss
                        print("save", valid_loss)
                        
    with tf.Session(graph=graph) as sess:
        saver.restore(sess=sess, save_path=save_path)
        feed_dict = {input_image: my_images}
        result = sess.run(tf_prediction,
                          feed_dict=feed_dict)
        result = np.argmax(result, axis=1)
    return result
                    
