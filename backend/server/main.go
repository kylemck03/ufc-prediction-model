package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type FighterStats struct {
	F1                  int     `json:"f_1"`
	NumRounds           int     `json:"num_rounds"`
	TitleFight          int     `json:"title_fight"`
	WeightClass         int     `json:"weight_class"`
	Gender              int     `json:"gender"`
	FighterHeightCm     float64 `json:"fighter_height_cm"`
	FighterWeightLbs    float64 `json:"fighter_weight_lbs"`
	FighterReachCm      float64 `json:"fighter_reach_cm"`
	FighterStance       float64 `json:"fighter_stance"`
	FighterW            int     `json:"fighter_w"`
	FighterL            int     `json:"fighter_l"`
	FighterD            int     `json:"fighter_d"`
	FighterDob          float64 `json:"fighter_dob"`
	AvgCtrlTime         float64 `json:"avg_ctrl_time"`
	AvgReversals        float64 `json:"avg_reversals"`
	AvgSubmissionAtt    float64 `json:"avg_submission_att"`
	AvgTakedownSucc     float64 `json:"avg_takedown_succ"`
	AvgTakedownAtt      float64 `json:"avg_takedown_att"`
	AvgSigStrikesAtt    float64 `json:"avg_sig_strikes_att"`
	AvgTotalStrikesSucc float64 `json:"avg_total_strikes_succ"`
	AvgTotalStrikesAtt  float64 `json:"avg_total_strikes_att"`
	AvgKnockdowns       float64 `json:"avg_knockdowns"`
	AvgFinishTime       float64 `json:"avg_finish_time"`
}

type PredictionRequest struct {
	Fighter1 FighterStats `json:"fighter1"`
	Fighter2 FighterStats `json:"fighter2"`
}

type PredictionResponse struct {
	Winner              string  `json:"winner"`
	Probability         float64 `json:"probability"`
	Fighter1Probability float64 `json:"fighter1_probability"`
	Fighter2Probability float64 `json:"fighter2_probability"`
}

func main() {
	r := gin.Default()

	//Enable CORS
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:3000"} // React App URL
	config.AllowMethods = []string{"POST", "GET"}
	r.Use(cors.New(config))

	// Routes
	r.POST("/predict", predictFight)
	r.Run(":8080") // run server
}

func predictFight(c *gin.Context) {
	var request PredictionRequest
	if err := c.BindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Log the request data
	fmt.Printf("Received request data: %+v\n", request)

	// Forward the request to the Python FastAPI service
	jsonData, err := json.Marshal(request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to marshal request"})
		return
	}

	// Log the JSON being sent
	fmt.Printf("Sending JSON to ML service: %s\n", string(jsonData))

	resp, err := http.Post("http://localhost:8001/predict", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to communicate with prediction service"})
		return
	}
	defer resp.Body.Close()

	// Log the response status
	fmt.Printf("ML service response status: %d\n", resp.StatusCode) // Add this

	var prediction PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse prediction response"})
		return
	}

	fmt.Printf("Final prediction: %+v\n", prediction)
	c.JSON(http.StatusOK, prediction)
}
