def training(train_path, valid_path, N_train, N_valid, img_width, img_height):
    
    train_data_dir = train_path
    validation_data_dir = valid_path
    nb_train_samples = N_train
    nb_validation_samples = N_valid
    epochs = 30
    batch_size = 16
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', 
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
                    rescale=1./255, 
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
    test_datagen = ImageDataGenerator(
                    rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='binary')
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save('model.h5')
def testing(model_path, img_path, img_width=70, img_height=70):

    model = load_model(model_path)
    img = load_img(img_path, False, target_size = (img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    preds = model.predict_classes(x)
    prob = model.predict_proba(x)
    print('preds: ' , preds)
    print('probs: ', prob)
    if preds[0][0] == 1:
        return 'class_1'
    else:
        return 'class_2'
 
