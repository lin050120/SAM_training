# SAM3 学習の一時停止と再開

## 機能の意味

これは永続的な一時停止です。学習プロセスを終了して GPU メモリを解放し、後で同じ run の
`checkpoints/checkpoint.pt` から再開します。UI や PC を再起動した後でも使用できます。

SAM3 は完了した各 epoch の終了後に `checkpoint.pt` を更新します。checkpoint にはモデル、
optimizer、epoch、step 数、loss 状態、AMP scaler が含まれます。epoch の途中で停止した場合、
最新 checkpoint より後の進捗は保持されません。

## 学習の一時停止

1. `学習プリフライト` ページを開きます。
2. モニターに `<run_dir>/checkpoints/checkpoint.pt` が表示されていることを確認します。
3. `学習を一時停止して GPU メモリを解放` をクリックします。
4. 状態が `paused` になるまで待ちます。学習プロセスは終了し、GPU メモリは解放されています。

最初の完全な checkpoint がまだない場合、再開不能な状態を防ぐため UI は一時停止を拒否します。
`checkpoint.pt.tmp` の保存中は、保存完了後にもう一度実行してください。

## 学習の再開

1. 同じページの `ステージ C: 最新の完全な Checkpoint から再開` を開きます。
2. `再開可能な run を更新` をクリックします。
3. 元の学習 run を選択します。
4. `再開プリフライト情報` で `resumable=true`、データパス、prompt、max epochs、
   `resume_checkpoint` が正しいことを確認します。
5. 再開確認チェックボックスをオンにして、`学習を再開` をクリックします。

再開時は元の run の runtime YAML、出力ディレクトリ、`checkpoint.pt` を再利用し、分散通信用の
port だけを新しく割り当てます。trainer checkpoint を `initial checkpoint` に入力しないでください。
その欄は新規 run の初期化用で、optimizer や epoch は復元しません。

## サーバー側のガード

再開前に次を再検証します。

- run、runtime YAML、空でない `checkpoint.pt` が存在する；
- 元の base `sam3.pt` と BPE ファイルが存在する；
- runtime YAML の checkpoint 出力先が選択した run 内である；
- train/validation パスが存在し、現在の dataset registry が元の学習モードを許可する；
- CUDA 数が元の run の要件を満たす；
- SAM301 trainer patch と SAM3 import パスが正しい；
- 同じ runtime YAML を使用中のプロセスが存在しない；
- 完了済み run を誤って再開しない。

`training_summary.json` のトップレベルは最新の起動結果です。`attempts` 配列には最初の起動、
一時停止、全ての再開履歴が同じ run 内に保存されます。

## 制限

- 未完了 epoch は保持されません。
- 最初の checkpoint 生成前は一時停止できません。
- 再開時に max epochs、batch size、learning rate、prompt、データセットは変更できません。
  変更が必要な場合は、新しいプリフライトと run を作成してください。
- 実行中 UI は更新したソースを自動では読み込みません。現在の学習が終わってから UI を再起動してください。
  旧 UI を閉じると、その UI が管理している活動中の学習も停止します。
