from src.task10_generation import INSUFFICIENT_EVIDENCE, generate_with_citation


def test_mandatory_rehab_cases_answerable():
    result = generate_with_citation("Những trường hợp nào phải đi cai nghiện bắt buộc?")
    answer = result["answer"].lower()
    assert result["answer"] != INSUFFICIENT_EVIDENCE
    assert "không đăng ký" in answer
    assert "tự ý chấm dứt" in answer


def test_huu_tin_case_answerable():
    result = generate_with_citation("Hữu Tín bị xử lý liên quan đến ma túy như thế nào?")
    answer = result["answer"].lower()
    assert result["answer"] != INSUFFICIENT_EVIDENCE
    assert "7 năm 6 tháng" in answer
    assert "tổ chức sử dụng trái phép chất ma túy" in answer


def test_famous_people_summary_covers_multiple_articles():
    result = generate_with_citation(
        "Những người nổi tiếng nào được nhắc đến trong dữ liệu liên quan đến ma túy?"
    )
    answer = result["answer"].lower()
    assert result["answer"] != INSUFFICIENT_EVIDENCE
    assert "chi dân" in answer
    assert "chu bin" in answer
