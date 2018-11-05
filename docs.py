import pdftotree


def main():

    pdftotree.parse(pdf_file='exampleReport.pdf', html_path='h/', model_type='ml', model_path='/home/fvoyager/Загрузки/data/model.pkl', favor_figures=True, visualize=False)


if __name__ == '__main__':
    main()