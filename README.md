# souko

## docs生成

1. docs内で以下を実行

```
sphinx-apidoc -f -o ../docs/source ../souko && make html
```

2. soukoリポジトリ内で下記を実行

```
sphinx-build -b html docs/source ~/git/souko-docs/latest
```