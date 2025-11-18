import runpy
import sys

sys.path.insert(0, '.')

def main():
    try:
        print('Running tests/test_pipeline.py ...')
        runpy.run_path('tests/test_pipeline.py', run_name='__main__')
        print('TESTS RAN: OK')
    except Exception as e:
        print('TESTS RAN: FAILED')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
