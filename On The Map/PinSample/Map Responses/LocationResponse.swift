//
//  LocationResponse.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 08/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct LocationResponse: Codable {
    let createdAt: String
    let firstName: String
    let lastName: String
    let latitude: Float
    let longitude: Float
    let mapString: String
    let mediaURL: String
    let objectId: String
    let uniqueKey: String
    let updatedAt: String

}
