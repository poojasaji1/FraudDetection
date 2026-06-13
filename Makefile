.PHONY: setup services data features train promote rollback shadow

setup:
	python -m pip install -r requirements.txt

services:
	docker-compose up -d

data:
	python data/validate.py

features:
	python features/feature_views.py
	cd features && feast apply
	python features/materialize.py

train:
	python training/train.py

promote:
	python registry/promote.py

rollback:
	python registry/rollback.py

shadow:
	python registry/shadow.py
