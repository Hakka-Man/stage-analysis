### Update yml
```
conda env export --from-history --name stage > stage.yml
```

### Update conda env with yml
```
conda env update --file stage.yml --prune
```