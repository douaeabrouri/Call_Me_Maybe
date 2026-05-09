

install:
    pip install -r requirements.txt

run:
    python3 call_me_maybe.py

debug:
    python3 -m db call_me_maybe.py

clean:
    rm -rf __pycach

act:
    source call_env/bin/activate

deact:
    deactivate

lint:
    flake8 . || exit 0
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-def --check-untyped-defs || exit 0

lint-strict:
    flake8 .
	mypy . --strict