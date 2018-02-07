load-rcp:
	wget http://agence-prd.ansm.sante.fr/php/ecodex/telecharger/fic_cis_cip.zip
	unzip fic_cis_cip.zip -d ./data/rcp
	rm fic_cis_cip.zip

clean: 
	rm -rf ./data/rcp fic_cis_cip.zip
