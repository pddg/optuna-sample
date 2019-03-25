import argparse
import functools
import chainer
import numpy as np
import optuna
from chainer import links as L
from chainer import functions as F
from chainer import training
from chainer.training import extensions

# From: https://github.com/chainer/chainer/blob/v5/examples/mnist/train_mnist.py
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# 目的関数を設定する
def objective(trial, device, train_data, test_data, prune):
    # trialからパラメータを取得
    n_unit = trial.suggest_int("n_unit", 8, 128)
    batch_size = trial.suggest_int("batch_size", 2, 128)
    n_out = 10
    epoch = 20

    # モデルを定義
    model = L.Classifier(MLP(n_unit, n_out))

    if device >= 0:
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size,
                                                 repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(
                    train_iter, optimizer, device=device)
    early_trigger = training.triggers.EarlyStoppingTrigger(
        check_trigger=(1, "epoch"),
        monitor="validation/main/accuracy",
        patients=3,
        mode="max",
        max_trigger=(epoch, "epoch")
    )
    trainer = training.Trainer(updater, early_trigger, out='output')

    # 実行中のログを取る
    log_reporter = extensions.LogReport()
    trainer.extend(log_reporter)

    # validationをするextensionを追加
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    # Optunaとのインテグレーションのためのextensionを追加
    # trialオブジェクト，監視するメトリクス，監視する頻度を指定
    if prune:
        integrator = optuna.integration.ChainerPruningExtension(
            trial, 'validation/main/accuracy', (1, 'epoch')
        )
        trainer.extend(integrator)

    # 学習を実行
    trainer.run()

    # Accuracyが最大のものを探す
    observed_log = log_reporter.log
    observed_log.sort(key=lambda x: x['validation/main/accuracy'])
    best_epoch = observed_log[-1]

    # 何epoch目がベストだったかを記録しておく
    trial.set_user_attr('epoch', best_epoch['epoch'])

    # accuracyを評価指標として用いる
    return 1 - best_epoch['validation/main/accuracy']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trials', type=int, help='Number of trials')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='GPU ID')
    parser.add_argument('--prune-with', choices=['median', 'asha', 'none'], default='none', help='Pruning method')
    args = parser.parse_args()

    np.random.seed(0)

    # MNISTデータを読み込む
    train, test = chainer.datasets.get_mnist()
    # 目的関数にパラメータを渡す
    obj = functools.partial(
            objective, device=args.gpu, train_data=train, 
            test_data=test, prune=True if args.prune_with != 'none' else False
    )

    # Prunerを作成
    if args.prune_with == 'none':
        pruner = None
    elif args.prune_with == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif args.prune_with == 'asha':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    # Studyを作成
    study = optuna.study.create_study(
        storage='sqlite:///optimize-{}.db'.format(args.prune_with), pruner=pruner, study_name='prune_test', load_if_exists=True
    )
    # 最適化を実行
    study.optimize(obj, n_trials=args.trials)

    # Summaryを出力
    print("[Trial summary]")
    df = study.trials_dataframe()
    state = optuna.structs.TrialState
    print("Copmleted:", len(df[df['state'] == state.COMPLETE]))
    print("Pruned:", len(df[df['state'] == state.PRUNED]))
    print("Failed:", len(df[df['state'] == state.FAIL]))

    # 最良のケース
    print("[Best Params]")
    best = study.best_trial
    print("Epoch:", best.user_attrs.get('epoch'))
    print("Accuracy:", 1 - best.value)
    print("Batch size:", best.params['batch_size'])
    print("N unit:", best.params['n_unit'])

