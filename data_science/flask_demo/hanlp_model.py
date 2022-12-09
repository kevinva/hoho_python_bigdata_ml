import hanlp

print(hanlp.pretrained.mtl.ALL)

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
HanLP('联通和腾讯成立新子公司', tasks = 'tok').pretty_print()
