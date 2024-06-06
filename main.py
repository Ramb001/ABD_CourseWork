from src.handlers import get_reviews, mendelbort, zipf_first_law, zipf_second_law
from model.model import Model


def main():
    url = str(input("\nВведите сслыку на отзывы интересующего товара: "))
    print("\nИдёт получение данных ...")
    reviews = get_reviews(url)

    print("\nModel initialization!\n")
    model = Model()
    model.training_model()
    model.predict(reviews)

    mendelbort(reviews)
    zipf_first_law(reviews)
    zipf_second_law(reviews)


if __name__ == "__main__":
    main()
