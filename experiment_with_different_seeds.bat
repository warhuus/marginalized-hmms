call env\Scripts\activate
for %%l in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001) do (
for %%s in (0, 112, 250) do (
echo %%l %%s
call env\Scripts\python bin\run.py --exp hard-dummy-data --algo direct --lrate %%l --seed %%s
call env\Scripts\python bin\run.py --exp hard-dummy-data --algo viterbi --lrate %%l --seed %%s
call env\Scripts\python bin\vis.py --num 2
)
)