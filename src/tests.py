import pytest
from wds import convert_lat_to_meters, convert_meters_to_lat, convert_long_to_meters, convert_meters_to_long
from wds import EARTH_CIRCUMFERENSE_ALONG_EQUATOR_KM, EARTH_CIRCUMFERENSE_ALONG_MERIDIAN_KM


class TestCoordinatesConverter:
    def test_convert_lat_to_meters_zero(self):
        print('тривиальный случай')
        assert convert_lat_to_meters(0) == 0

    def test_convert_lat_to_meters_perimeter_around_poles(self):
        print('окружность Земли вокруг полюсов')
        half_of_meridian_m = convert_lat_to_meters(90.0)
        assert half_of_meridian_m * 4 == EARTH_CIRCUMFERENSE_ALONG_MERIDIAN_KM * 1000

    def test_convert_lat_to_meters_range(self):
        print('проверка граничных значений')
        # assert convert_lat_to_meters(90.1) == 0
        with pytest.raises(IndexError) as exc:
            convert_lat_to_meters(90.1)
        assert 'ValueError' == str(exc.value)

    def test_convert_meters_to_lat(self):
        print('тривиальный случай')
        assert convert_meters_to_lat(0) == 0

    def test_convert_meters_to_lat_perimeter_around_poles(self):
        print('окружность Земли вокруг полюсов')
        assert convert_meters_to_lat(EARTH_CIRCUMFERENSE_ALONG_MERIDIAN_KM * 1000 / 4) == 90

    def test_convert_long_to_meters_zero(self):
        print('тривиальный случай')
        assert convert_long_to_meters(0, 0) == 0
        assert convert_long_to_meters(0, 45) == 0
        assert convert_long_to_meters(0, 90) == 0

    def test_convert_lat_to_meters_perimeter_around_equator(self):
        print('окружность Земли вокруг экватора')
        half_of_meridian_m = convert_long_to_meters(90.0, 0)
        assert half_of_meridian_m * 4 == EARTH_CIRCUMFERENSE_ALONG_EQUATOR_KM * 1000

