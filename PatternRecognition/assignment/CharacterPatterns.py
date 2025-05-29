# 패턴인식 - 특징 벡터 기반 문자 분류기 (트리 구조 구현)

# ◆ 1. 각 문자의 특징 벡터 정의
# 특징1: 수직선 수(V)
# 특징2: 수평선 수(H)
# 특징3: 기울어진 직선 수(O)
# 특징4: 커브 수(C)
feature_dict = {
    "L": [1, 1, 0, 0],
    "P": [1, 0, 0, 1],
    "O": [0, 0, 0, 1],
    "E": [1, 3, 0, 0],
    "Q": [0, 0, 1, 1]
}


# ◆ 2. 트리 구조 분류 함수 정의
def classify_character(features):
    V, H, O, C = features  # 특징 벡터 분해

    if V > 0:
        if C > 0:
            return "P"
        else:
            if H > 0:
                return "E"
            else:
                return "L"
    else:
        if O > 0:
            return "Q"
        else:
            return "O"


# ◆ 3. 테스트 실행 (모든 문자에 대해 분류 결과 확인)
print("=== 문자 분류기 테스트 결과 ===")
for actual_char, feature_vec in feature_dict.items():
    predicted_char = classify_character(feature_vec)
    print(f"실제 문자: {actual_char}  →  분류 결과: {predicted_char}")