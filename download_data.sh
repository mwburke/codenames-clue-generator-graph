#!/bin/bash
mkdir -p data
# Use -C - to resume if it fails partway
curl -C - -o data/enwiki-latest-page.sql.gz https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz
curl -C - -o data/enwiki-latest-pagelinks.sql.gz https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pagelinks.sql.gz
curl -C - -o data/enwiki-latest-redirect.sql.gz https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz
curl -C - -o data/enwiki-latest-linktarget.sql.gz https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-linktarget.sql.gz
