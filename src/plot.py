from wds import WheatDetectionSystem

wds = WheatDetectionSystem('Field2_3_2019', '07_25')
wds.create_mask('tgi')
# wds.read_bboxes('frcnn', 512)
# wds.perform_calculations()
# wds.draw_wheat_plots()

wds.create_map()
wds.draw_protocol()
wds.draw_tiles('src')
wds.draw_tiles('tgi')
wds.draw_vegetation()
wds.save_map()
