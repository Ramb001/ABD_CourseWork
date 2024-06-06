import json
import time
import math

import undetected_chromedriver as uc
from bs4 import BeautifulSoup

from collections import Counter
import matplotlib.pyplot as plt

from .contants import ReviewsType


def get_reviews(url: str) -> list[ReviewsType]:
    driver = uc.Chrome(version_main=124)
    driver.get(url)
    time.sleep(7)
    html_sourse_code = driver.execute_script("return document.body.innerHTML;")
    soup = BeautifulSoup(html_sourse_code, "html.parser")
    driver.quit()
    return [
        {
            "review": f'{review["content"]["comment"]}\n{review["content"]["positive"]}\n{review["content"]["negative"] if review["content"]["negative"].lower() != ("нет" or "не обнаружил" or "") else ""}',
            "score": str(review["content"]["score"]),
        }
        for review in json.loads(
            str(soup.find_all("div", id="state-webListReviews-3231710-default-1")[0])[
                17:-52
            ]
        )["reviews"]
    ]


def mendelbort(reviews: list[ReviewsType]):
    counter = {
        k: v
        for k, v in sorted(
            dict(
                Counter([word for item in reviews for word in item["review"].split()])
            ).items(),
            key=lambda x: x[1],
        )
    }
    res = []
    diagram_data = []

    for k, v in counter.items():
        temp = math.fabs(math.log(len(k), (1 * 0.1) / v))
        res.append(temp)
        if temp >= 0.5:
            diagram_data.append({k: temp})

    print(f"Естественность Языка: {round(sum(res) / len(res), 5)} из 1")

    diagram_data = diagram_data[:10]
    _, ax = plt.subplots()
    ax.pie(
        [value for item in diagram_data for _, value in item.items()],
        labels=[key for item in diagram_data for key, _ in item.items()],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.grid()
    plt.show()


def zipf_first_law(reviews: list[ReviewsType]):
    counter = {
        k: v
        for k, v in sorted(
            dict(
                Counter([word for item in reviews for word in item["review"].split()])
            ).items(),
            key=lambda x: x[1],
        )
    }

    keys_length = {}

    for key in counter.keys():
        keys_length[key] = len(key)

    sorted_key_length = {
        k: v for k, v in sorted(keys_length.items(), key=lambda item: item[1])
    }

    plt.title("Первый закон Ципфа")
    plt.xlabel("Значение слова")
    plt.ylabel("Частота")
    plt.plot(
        list(sorted_key_length.values()),
        [counter[key] for key in list(sorted_key_length.keys())],
    )
    plt.show()


def zipf_second_law(reviews: list[ReviewsType]):
    counter = dict(
        Counter([word for item in reviews for word in item["review"].split()])
    )

    lengths = [length for length in range(len(counter))]

    plt.title("Второй закон ципфа")
    plt.xlabel("Частота")
    plt.ylabel("Количество слов")
    plt.plot(lengths, list(counter.values()))
    plt.show()
