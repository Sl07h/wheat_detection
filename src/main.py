from wds import WheatDetectionSystem

wds = WheatDetectionSystem('test_set1', '07_25', 'frcnn', '400')
wds.read_metadata()
wds.read_bboxes()
wds.perform_calculations()

wds.create_map()
wds.draw_wheat_plots()
wds.draw_protocol()
wds.draw_grids([0.25, 0.5])
wds.save_map()
