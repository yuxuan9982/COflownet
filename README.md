This is an implementation of COFlownet with four different tasks.

The first one is grid, to run this code, you can simply run.

```
cd grid/
python conservative.py --dataType=mixed.pt
```

The second one is molecule design, to run our offline version, 

```
cd mols/
python conservative-gflownet.py
```

The Third one is bioseq design, to run our offline version, 

```
cd bioseq/
python offline-amp.py
```

The Last one is item recommendation, to run our offline version, you can simply run our scripts with default setting. For a more detailed analysis, you can find all available scripts in the `recommend/scripts/` folder and customize them according to your needs.

```
cd recommend/
sh simple_run.sh
```

