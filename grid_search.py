import matplotlib.pyplot as plt
import classifier_definitions
import create_dataset
from tensorflow import keras


colors = ['b', 'g', 'r', 'c', 'm', 'y']


def run_grid_search(model_name, layer_options, dropout_options, lr_options, reg_options, batch_options, skip_num):
    dataLoader = create_dataset.DataLoader()
    trainSet = dataLoader.getDataset(num_samples=300, normalize=True)[0]  # gives first 30 subjects
    testSet = dataLoader.getDataset(start_index=300, normalize=True)[0]  # gives last 6 subjects

    # Apply transformations to expand dataset
    trainSet = classifier_definitions.applyTransform(trainSet, num_repeat=5)

    # If grid search is somehow halted, you can use count to specify a number of hyperparameter
    # values to skip and resume the search
    count = skip_num
    skip = True

    stopCallback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)

    if model_name == 'svm':
        for layer in layer_options:
            for drop in dropout_options:
                for lr in lr_options:
                    for reg in reg_options:
                        for batch in batch_options:
                            if skip:
                                if count != 0:
                                    count = count - 1
                                    continue
                                else:
                                    skip = False

                            current_model = classifier_definitions.get_generalizedSVM(num_conv_layers=layer, dropout=drop, learning_rate=lr,
                                                           regularizer=reg)
                            current_trainSet = trainSet.shuffle(200).batch(batch)
                            current_testSet = testSet.batch(batch)
                            history = current_model.fit(current_trainSet, epochs=100, validation_data=current_testSet,
                                                        callbacks=[stopCallback])

                            fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
                            plt.plot(history.history['accuracy'], label="train acc")
                            plt.plot(history.history['val_accuracy'], label="test acc")
                            plt.plot(history.history['top3_acc'], label="train top3")
                            plt.plot(history.history['val_top3_acc'], label="test top3")
                            plt.title(
                                f"{history.history['val_accuracy'][-1]:.4f}% accuracy generalizedSVM: layers={layer}, dropout={drop}, lr={lr}, reg={reg}, batch={batch}")
                            plt.xlabel("epochs")
                            plt.ylabel("accuracy")
                            plt.legend()
                            plt.savefig(
                                f"./figures/{history.history['val_accuracy'][-1]:.4f}GeneralizedSVM#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}.png")
                            plt.close()

    elif model_name == 'cnn':
        trainSet = classifier_definitions.reduceSize(trainSet)
        testSet = classifier_definitions.reduceSize(testSet)
        testSet = testSet.batch(10)

        for layer in layer_options:
            for drop in dropout_options:
                for lr in lr_options:
                    for batch in batch_options:
                        print(f"cnn2#l{layer}-d{drop}-lr{lr}-b{batch}\n\n")
                        if skip:
                            if count != 0:
                                count = count - 1
                                continue
                            else:
                                skip = False
                        current_model = classifier_definitions.get_CNN(num_conv_layer_groups=layer, dropout=drop, learning_rate=lr)
                        current_trainSet = trainSet.shuffle(200).batch(batch)
                        history = current_model.fit(current_trainSet, epochs=65, validation_data=testSet, callbacks=[stopCallback])

                        fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
                        plt.plot(history.history['accuracy'], label="train acc")
                        plt.plot(history.history['val_accuracy'], label="test acc")
                        plt.plot(history.history['top3_acc'], label="train top3")
                        plt.plot(history.history['val_top3_acc'], label="test top3")
                        plt.title(f"{history.history['val_accuracy'][-1]:.4f}% accuracy cnn2: layers={layer}, dropout={drop}, lr={lr}, batch={batch}")
                        plt.xlabel("epochs")
                        plt.ylabel("accuracy")
                        plt.legend()
                        plt.savefig(f"./figures/{history.history['val_accuracy'][-1]:.4f}cnn2#l{layer}-d{drop}-lr{lr}-b{batch}.png")
                        plt.close()


def evaluate_best_models(model_type, num_models, layers, dropouts, lrs, regs, batches, epochs, patience):
    fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
    plt.title(f"{num_models} Most Accurate {model_type}'s")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    stopCallback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)

    # retest the model
    dataLoader = create_dataset.DataLoader()
    trainSet = dataLoader.getDataset(num_samples=300, normalize=True)[0]  # gives first 30 subjects
    testSet = dataLoader.getDataset(start_index=300, normalize=True)[0]  # gives last 6 subjects

    # Apply transformations to expand dataset
    trainSet = classifier_definitions.applyTransform(trainSet, num_repeat=5)

    if model_type == 'CNN':
        trainSet = classifier_definitions.reduceSize(trainSet)
        testSet = classifier_definitions.reduceSize(testSet)

    testSet = testSet.batch(10)

    for i in range(0, num_models):
        layer = layers[i]
        drop = dropouts[i]
        lr = lrs[i]
        batch = batches[i]

        if model_type == 'CNN':
            current_model = classifier_definitions.get_CNN(num_conv_layer_groups=layer, dropout=drop, learning_rate=lr)
            current_trainSet = trainSet.shuffle(200).batch(batch)
            history = current_model.fit(current_trainSet, epochs=epochs, validation_data=testSet, callbacks=[stopCallback])

            # save the model with the highest accuracy weights

            # add to the plot
            plt.plot(history.history['val_accuracy'], label=f"{model_type}#l{layer}-d{drop}-lr{lr}-b{batch}: test acc",
                     color=colors[i])
            plt.plot(history.history['val_top3_acc'], label=f"{model_type}#l{layer}-d{drop}-lr{lr}-b{batch}: test top3",
                     color=colors[i], linestyle='dashed')
            current_model.save(f"./trained_models/{model_type}-{i}#l{layer}-d{drop}-lr{lr}-b{batch}.h5")

        elif model_type == "SVM":
            reg = regs[i]

            current_model = classifier_definitions.get_generalizedSVM(num_conv_layers=layer, dropout=drop,
                                                                      learning_rate=lr,
                                                                      regularizer=reg)
            current_trainSet = trainSet.shuffle(200).batch(batch)
            current_testSet = testSet.batch(10)
            history = current_model.fit(current_trainSet, epochs=epochs, validation_data=testSet,
                                        callbacks=[stopCallback])

            # add to the plot
            plt.plot(history.history['val_accuracy'], label=f"{model_type}#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}: test acc",
                     color=colors[i])
            plt.plot(history.history['val_top3_acc'], label=f"{model_type}#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}: test top3",
                     color=colors[i], linestyle='dashed')
            current_model.save(f"./trained_models/{model_type}-{i}#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}.h5")

        else:
            return

    plt.legend()
    plt.savefig(f"./figures/Best {num_models} {model_type}'s.png")
    plt.close()


if __name__ == '__main__':
    # perform a grid search
    # layer_options = [2, 3, 4]
    # dropout_options = [0.2, 0.3, 0.4, 0.5]
    # lr_options = [0.1, 0.01, 0.001]
    # reg_options = [1, 0.1, 0.01]
    # batch_options = [15, 30, 60]
    # skip = 226
    #
    # run_grid_search('svm', layer_options, dropout_options, lr_options, reg_options, batch_options, skip)
    # # run_grid_search('cnn', layer_options, dropout_options, lr_options, reg_options, batch_options, skip)

    # compare the best models
    # layer_options = [2, 2, 2, 2]
    # dropout_options = [0.2, 0.4, 0.3, 0.5]
    # lr_options = [0.001, 0.001, 0.001, 0.001]
    # reg_options = [0.1, 0.01]
    # batch_options = [60, 60, 60, 60]
    #
    # evaluate_best_models('CNN', 4, layer_options, dropout_options, lr_options, reg_options, batch_options, epochs=200, patience=20)

    layer_options = [3, 3, 3, 3]
    dropout_options = [0.2, 0.4, 0.5, 0.3]
    lr_options = [0.001, 0.001, 0.001, 0.001]
    reg_options = [0.1, 0.01, 0.01, 0.1]
    batch_options = [30, 30, 60, 60]

    evaluate_best_models('SVM', 4, layer_options, dropout_options, lr_options, reg_options, batch_options, epochs=200, patience=20)

