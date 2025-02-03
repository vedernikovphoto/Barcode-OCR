install:
	pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

train:
	export PYTHONPATH=$(PWD) && python src/train.py configs/config.yaml

inference:
	export PYTHONPATH=$(PWD) && python src/inference.py

download:
	@echo "Fetching download URL..."
	@wget -q -O - 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/pRFNuxLQUZcDDg' \
	| grep -oP '(?<="href":")[^"]*' > download_url.txt
	@echo "Downloading dataset..."
	@wget -O dataset.zip -i download_url.txt
	@rm download_url.txt
	@echo "Download complete."
	@echo "Extracting files..."
	python -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('.')"
	rm dataset.zip
	@if [ -f data/README.md ]; then rm data/README.md; fi
	@if [ -d data/images ]; then mv data/images data/original_images; fi
	@echo "Copy df_train.csv and df_val.csv to the data folder..."
	@cp df_train.csv data/
	@cp df_valid.csv data/
	@echo "Preprocessing images..."
	export PYTHONPATH=$(PWD) && python src/crop_and_resize_gt.py